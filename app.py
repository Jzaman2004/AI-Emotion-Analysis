import base64
import json
import math
import logging
import os
from io import BytesIO
from typing import Dict, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from groq import Groq
from PIL import Image, UnidentifiedImageError

load_dotenv()

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
EMOTION_KEYS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "hate",
    "confusion",
    "frustration",
    "boredom",
    "contempt",
]

SYSTEM_PROMPT = (
    "You are an ethical facial analysis assistant. Analyze the provided facial image and output ONLY a valid JSON object with the following structure. Do not include any other text, explanations, or markdown.\n"
    '{"emotions": {"neutral": int, "joy": int, "sadness": int, "anger": int, "fear": int, "disgust": int, "surprise": int, "hate": int, "confusion": int, "frustration": int, "boredom": int, "contempt": int}, "dominant_emotion": string}\n'
    "RULES: 1. All 12 emotion percentages MUST sum to exactly 100. 2. dominant_emotion must be the key with the highest integer value. If tie, pick first alphabetically. 3. If confidence is low, distribute evenly or default to neutral — but still sum to 100. 4. Output ONLY the JSON object."
)

EXPLANATION_PROMPT = (
    "You are an ethical facial expression analyst. Based on the same image and the dominant emotion label, explain why the image likely matches that emotion. "
    "Be concrete and image-based: name the visible facial cues you can infer, such as mouth shape, eyebrow position, eye openness, tears, cheek tension, jaw tension, forehead tension, gaze direction, or overall expression, and explain how those cues support the label. "
    "Do not mention percentages, probabilities, model internals, or generic filler like 'the model is likely.' Write as a direct explanation of the visible face. Keep it to 2-3 short sentences. Return ONLY a valid JSON object with this structure: "
    '{"analysis_explanation": string}'
)

EXPLANATION_RETRY_PROMPT = (
    "You are rewriting a weak facial-expression explanation. The output must describe only the visible evidence in the image that supports the dominant emotion. "
    "Use concrete facial cues such as eyebrows, eyes, tears, mouth shape, jaw tension, cheeks, forehead, or gaze. "
    "Do not mention scores, percentages, ranks, confidence, or generic summary phrases like 'came out on top' or 'supporting signals.' "
    "Write exactly 2 short sentences that sound like image observation, not analysis metadata. Return ONLY a valid JSON object with this structure: "
    '{"analysis_explanation": string}'
)

SUMMARY_PHRASES = (
    "came out on top",
    "supporting signals",
    "strongest relative signal",
    "mixed expression",
    "result leans",
    "relative pattern",
    "best-fit label",
    "next at",
    "follows at",
)


def explanation_looks_like_score_summary(text: str) -> bool:
    lowered = text.lower()
    if "%" in lowered:
        return True
    return any(phrase in lowered for phrase in SUMMARY_PHRASES)


def request_analysis_explanation(client: Groq, image_data_url: str, dominant_emotion: str) -> str:
    prompts = [EXPLANATION_PROMPT, EXPLANATION_RETRY_PROMPT]

    for attempt, prompt in enumerate(prompts, start=1):
        explanation_completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=280,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"The dominant emotion is {dominant_emotion}. Give the visible facial evidence that explains why this image matches that label. "
                                "Make the answer sound like an observation of the face itself."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        )

        explanation_content = (explanation_completion.choices[0].message.content or "").strip()
        explanation_json = parse_model_json(explanation_content)
        analysis_explanation = ""
        if isinstance(explanation_json, dict):
            analysis_explanation = explanation_json.get("analysis_explanation", "")
            if not isinstance(analysis_explanation, str):
                analysis_explanation = ""

        analysis_explanation = analysis_explanation.strip()
        if analysis_explanation and not explanation_looks_like_score_summary(analysis_explanation):
            return analysis_explanation

        app.logger.warning(
            "Explanation attempt %s returned score-summary style text; retrying with stricter prompt.",
            attempt,
        )

    raise ValueError("Model did not return a usable explanation for the dominant emotion.")


def error_response(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def compute_dominant_emotion(emotions: Dict[str, int]) -> str:
    max_value = max(emotions.values())
    candidates = [key for key, value in emotions.items() if value == max_value]
    return sorted(candidates)[0]


def normalize_emotions(raw_emotions: Dict[str, float]) -> Dict[str, int]:
    values = {}
    for key in EMOTION_KEYS:
        value = raw_emotions.get(key, 0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        values[key] = max(0.0, numeric)

    # Keep neutral in the schema, but force it to zero by design.
    values["neutral"] = 0.0
    non_neutral_keys = [key for key in EMOTION_KEYS if key != "neutral"]

    total = sum(values[key] for key in non_neutral_keys)
    if total <= 0:
        even = 100 / len(non_neutral_keys)
        scaled = {key: even for key in non_neutral_keys}
    else:
        scaled = {key: (values[key] / total) * 100 for key in non_neutral_keys}

    floored = {key: int(math.floor(scaled[key])) for key in non_neutral_keys}
    remainder = 100 - sum(floored.values())

    # Distribute the remainder to the largest fractional components first.
    ranking = sorted(
        non_neutral_keys,
        key=lambda k: (-(scaled[k] - floored[k]), -scaled[k], k),
    )

    for i in range(remainder):
        floored[ranking[i % len(ranking)]] += 1

    floored["neutral"] = 0

    return floored


def parse_model_json(content: str) -> Dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def image_bytes_to_data_url(image_bytes: bytes, image_format: str) -> str:
    mime = "image/jpeg" if image_format.upper() in {"JPG", "JPEG"} else "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def validate_image_bytes(image_bytes: bytes) -> Tuple[bytes, str]:
    if not image_bytes:
        raise ValueError("No image data provided.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise OverflowError("Image size exceeds 4MB limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            image_format = (img.format or "").upper()
            if image_format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported.")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image payload.") from exc

    return image_bytes, image_format


def extract_image_from_request() -> Tuple[bytes, str]:
    has_json = request.is_json
    has_files = bool(request.files)

    if has_json and has_files:
        raise ValueError("Provide either base64 JSON payload or one image file, not both.")

    if has_json:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image_data")
        if not image_data or not isinstance(image_data, str):
            raise ValueError("Missing image_data in JSON body.")
        if not image_data.startswith("data:image/") or ";base64," not in image_data:
            raise ValueError("image_data must be a valid base64 data URL.")

        header, encoded = image_data.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "").lower()
        if media_type not in {"image/jpeg", "image/jpg", "image/png"}:
            raise ValueError("Only JPEG and PNG images are supported.")

        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 image_data.") from exc

        return validate_image_bytes(image_bytes)

    if has_files:
        if len(request.files) != 1:
            raise ValueError("Exactly one image file is required.")

        file_storage = next(iter(request.files.values()))
        if not file_storage or not file_storage.filename:
            raise ValueError("No image file uploaded.")

        image_bytes = file_storage.read()
        return validate_image_bytes(image_bytes)

    raise ValueError("No image input found. Send JSON image_data or one multipart file.")


def analyze_with_groq(image_data_url: str) -> Dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this facial image."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )

    content = (completion.choices[0].message.content or "").strip()
    model_json = parse_model_json(content)
    raw_emotions = model_json.get("emotions") if isinstance(model_json, dict) else {}
    if not isinstance(raw_emotions, dict):
        raw_emotions = {}

    normalized = normalize_emotions(raw_emotions)
    dominant_emotion = compute_dominant_emotion(normalized)

    analysis_explanation = request_analysis_explanation(client, image_data_url, dominant_emotion)

    return {
        "emotions": normalized,
        "dominant_emotion": dominant_emotion,
        "analysis_explanation": analysis_explanation,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/results", methods=["GET"])
def results():
    return render_template("results.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if request.content_length and request.content_length > (MAX_IMAGE_BYTES + 1024 * 1024):
        return error_response("Request payload is too large.", 413)

    try:
        image_bytes, image_format = extract_image_from_request()
        image_data_url = image_bytes_to_data_url(image_bytes, image_format)
        model_output = analyze_with_groq(image_data_url)

        return jsonify(
            {
                "image_data": image_data_url,
                "emotions": model_output["emotions"],
                "dominant_emotion": model_output["dominant_emotion"],
                "analysis_explanation": model_output["analysis_explanation"],
            }
        )
    except OverflowError as exc:
        return error_response(str(exc), 413)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except RuntimeError as exc:
        return error_response(str(exc), 500)
    except json.JSONDecodeError:
        return error_response("Model response could not be parsed as valid JSON.", 500)
    except Exception as exc:
        app.logger.exception("Unexpected /analyze failure")
        return error_response(f"Internal server error while processing analysis: {str(exc)}", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
