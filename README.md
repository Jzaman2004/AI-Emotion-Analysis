# 🧠 AI Emotion Analysis: Benchmarking Transformer-Based Facial Emotion Recognition

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## 📖 Overview
This repository contains the complete codebase, statistical analysis pipeline, and results for my research on **benchmarking Transformer-Based AI (TB-AI) for facial emotion recognition**. As AI emotion recognition enters high-stakes domains (educational monitoring, digital safety), this study evaluates the reliability with which open-weight vision-language models classify 12 target emotions across basic and complex categories.

## 🎯 Research Objectives & Hypotheses
- **H1 (Basic vs. Complex Accuracy):** TB-AI will demonstrate significantly higher classification accuracy for basic, visually distinct emotions compared to complex, socially nuanced emotions.
- **H2 (Confidence Calibration):** Model confidence scores will positively correlate with classification correctness for basic emotions but show weak/null correlation for complex emotions.

## 📦 Dataset & Methodology
- **Sample:** 100 balanced facial images (8 per emotion × 12 emotions + 4 mixed)
- **Sources:** Open-source datasets + AI-generated expressions
- **Model:** Open-weight transformer VLM (`meta-llama/llama-4-scout-17b-16e-instruct`)
- **Pipeline:** Image ingestion → VLM inference → JSON parsing → Emotion probability distribution → Statistical evaluation

## 📈 Statistical Framework
| Analysis | Method | Output |
|----------|--------|--------|
| Per-Emotion Accuracy | 95% Clopper-Pearson Exact CI | `results/per_emotion_accuracy.csv` |
| Classification Performance | 13×12 Asymmetrical Confusion Matrix | `results/confusion_matrix_*.csv` |
| Hypothesis Testing (H1) | Mann-Whitney U Test | `results/hypothesis_test_results.csv` |
| Confidence Calibration (H2) | Point-Biserial Correlation | `results/confidence_calibration.csv` |

## 🔑 Key Findings
- **Overall Accuracy:** 46.0% across 100 images
- **Basic vs. Complex:** Basic emotions achieved 73.2% accuracy; complex emotions achieved 12.5% (`U = 1800.0, p < 0.001, r = -0.607`)
- **Emotion-Level Performance:** Neutral & Joy reached 100%; Hate, Boredom & Contempt reached 0%
- **Calibration:** Strong confidence-correctness correlation for 6/12 emotions (average `r_pb = 0.841, p < 0.05`)
- **Systematic Bias:** Complex emotions consistently misclassified into basic categories, revealing training data gaps

## 🌍 Ethical Considerations & Policy Implications
1. Current accuracy remains insufficient for high-stakes automated deployment.
2. Emotion-specific benchmarks are required prior to institutional adoption.
3. Demographic fairness audits remain essential across diverse populations.
4. Automated decision-making for complex emotions should be restricted until accuracy exceeds 70% with narrow confidence intervals.

## Demo App Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.
