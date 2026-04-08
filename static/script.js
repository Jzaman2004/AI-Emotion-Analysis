const MAX_DIMENSION = 1024;
const ANALYSIS_KEY = "emotion_analysis_result";
const THEME_KEY = "theme_mode";
const EMOTION_KEYS = [
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
];

function showToast(message) {
    const toast = document.getElementById("toast");
    if (!toast) return;

    toast.textContent = message;
    toast.classList.remove("hidden");
    window.setTimeout(() => toast.classList.add("hidden"), 3500);
}

function setLoading(isLoading) {
    const overlay = document.getElementById("loadingOverlay");
    if (!overlay) return;
    overlay.classList.toggle("hidden", !isLoading);
}

function normalizeErrorMessage(err) {
    if (typeof err === "string") return err;
    if (err && typeof err.message === "string") return err.message;
    return "Analysis failed. Please retry.";
}

function formatEmotionLabel(key) {
    return key
        .split("_")
        .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
        .join(" ");
}

function applyTheme(mode) {
    document.body.classList.toggle("dark-mode", mode === "dark");
    const toggle = document.getElementById("themeToggle");
    if (toggle) {
        toggle.textContent = mode === "dark" ? "Light Mode" : "Dark Mode";
    }
}

function initThemeToggle() {
    const saved = localStorage.getItem(THEME_KEY);
    const prefersDark =
        window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const current = saved || (prefersDark ? "dark" : "light");
    applyTheme(current);

    const toggle = document.getElementById("themeToggle");
    if (!toggle) return;

    toggle.addEventListener("click", () => {
        const next = document.body.classList.contains("dark-mode") ? "light" : "dark";
        localStorage.setItem(THEME_KEY, next);
        applyTheme(next);
    });
}

function resizeVideoFrameToDataUrl(video) {
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;

    if (!sourceWidth || !sourceHeight) {
        throw new Error("Camera feed is not ready.");
    }

    const scale = Math.min(1, MAX_DIMENSION / Math.max(sourceWidth, sourceHeight));
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.drawImage(video, 0, 0, width, height);

    return canvas.toDataURL("image/jpeg", 0.9);
}

async function initCapturePage() {
    const video = document.getElementById("cameraFeed");
    const captureBtn = document.getElementById("captureBtn");

    if (!video || !captureBtn) return;

    let mediaStream;

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
            audio: false,
        });
        video.srcObject = mediaStream;
        await video.play();
    } catch (err) {
        showToast("Unable to access webcam. Check browser permissions.");
        captureBtn.disabled = true;
        return;
    }

    captureBtn.addEventListener("click", async () => {
        if (captureBtn.disabled) return;

        captureBtn.disabled = true;
        setLoading(true);

        try {
            const imageData = resizeVideoFrameToDataUrl(video);

            if (mediaStream) {
                mediaStream.getTracks().forEach((track) => track.stop());
            }

            const response = await fetch("/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ image_data: imageData }),
            });

            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.error || "Analysis request failed.");
            }

            sessionStorage.setItem(ANALYSIS_KEY, JSON.stringify(payload));
            window.location.href = "/results";
        } catch (err) {
            showToast(normalizeErrorMessage(err));
            captureBtn.disabled = false;
            setLoading(false);
        }
    });
}

function renderEmotionRows(emotions) {
    const leftList = document.getElementById("emotionListLeft");
    const rightList = document.getElementById("emotionListRight");
    const legacyList = document.getElementById("emotionList");
    if (!leftList || !rightList) {
        if (!legacyList) return;
        legacyList.innerHTML = "";
    }

    if (leftList && rightList) {
        leftList.innerHTML = "";
        rightList.innerHTML = "";
    }
    const safeEmotions = {};
    for (const key of EMOTION_KEYS) {
        const value = Number(emotions?.[key]);
        safeEmotions[key] = Number.isFinite(value) ? Math.max(0, Math.min(100, Math.round(value))) : 0;
    }

    const entries = Object.entries(safeEmotions).sort((a, b) => {
        if (b[1] !== a[1]) return b[1] - a[1];
        return a[0].localeCompare(b[0]);
    });
    const midpoint = Math.ceil(entries.length / 2);

    for (let i = 0; i < entries.length; i += 1) {
        const [name, value] = entries[i];
        const row = document.createElement("div");
        row.className = "emotion-item";

        row.innerHTML = `
            <div class="emotion-meta">
                <span class="emotion-name">${formatEmotionLabel(name)}</span>
                <span class="emotion-value">${value}%</span>
            </div>
            <div class="progress-track" aria-hidden="true">
                <div class="progress-fill" style="width:${Math.max(0, Math.min(100, value))}%"></div>
            </div>
        `;

        if (!leftList || !rightList) {
            legacyList.appendChild(row);
        } else if (i < midpoint) {
            leftList.appendChild(row);
        } else {
            rightList.appendChild(row);
        }
    }
}

function initResultsPage() {
    const retryBtn = document.getElementById("retryBtn");
    const resultImage = document.getElementById("resultImage");
    const dominantEmotion = document.getElementById("dominantEmotion");
    const dominantPanel = document.getElementById("dominantPanel");

    const raw = sessionStorage.getItem(ANALYSIS_KEY);
    if (!raw) {
        window.location.href = "/";
        return;
    }

    let data;
    try {
        data = JSON.parse(raw);
    } catch {
        sessionStorage.removeItem(ANALYSIS_KEY);
        window.location.href = "/";
        return;
    }

    const { image_data: imageData, emotions, dominant_emotion: dominant } = data;
    if (!imageData || !emotions || !dominant) {
        sessionStorage.removeItem(ANALYSIS_KEY);
        window.location.href = "/";
        return;
    }

    resultImage.src = imageData;
    dominantEmotion.textContent = formatEmotionLabel(dominant);
    dominantPanel.classList.add("highlight");
    renderEmotionRows(emotions);

    if (retryBtn) {
        retryBtn.addEventListener("click", () => {
            sessionStorage.removeItem(ANALYSIS_KEY);
            window.location.href = "/";
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    const page = document.body.dataset.page;
    if (page === "capture") {
        initCapturePage();
    }
    if (page === "results") {
        initResultsPage();
    }
});
