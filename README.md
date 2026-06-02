# 🏥 Clinical Note Intelligence System

An end-to-end AI-powered clinical NLP pipeline that extracts structured medical information from unstructured clinical notes and predicts patient risk level.

---

## 📁 Directory Structure

```
clinical_note_intelligence/
│
├── models/
│   ├── risk_model.pkl           ← XGBoost risk classifier
│   ├── risk_encoder.pkl         ← Risk label encoder
│   ├── ohe_encoder.pkl          ← One-hot encoder for specialty
│   └── sentence_encoder.pkl     ← SentenceTransformer encoder
│
├── data/
│   └── mtsamples_final.csv      ← Processed dataset with all outputs
│
├── app.py                       ← Streamlit dashboard
├── pipeline.py                  ← 3-model inference pipeline
├── requirements.txt             ← Dependencies
└── README.md                    ← This file
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Add trained models
Copy your trained `.pkl` files into the `models/` folder.

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🧠 Pipeline Architecture

```
Transcription + Description
        ↓
Model 1: Abstractive Summarization
(sshleifer/distilbart-cnn-12-6)
        ↓
Model 2: Keyword Extraction
(KeyBERT + BioBERT embeddings)
        ↓
Model 3: Risk Prediction
(XGBoost + SentenceTransformer features)
        ↓
Risk Label: Low / Medium / High
```

---

## 📊 Model Performance

| Model | Task | Metric | Score |
|---|---|---|---|
| Model 1 | Summarization | ROUGE-2 | 0.29 |
| Model 2 | Keyword Extraction | Cosine Similarity | 0.62 |
| Model 3 | Risk Prediction | Accuracy | 0.92 |

---

## 📦 Dataset

- **Source**: MTSamples (medicaltranscriptions on Kaggle)
- **Size**: 4,921 clinical notes
- **Specialties**: 40+ medical specialties

---

---

## Link
https://clinical-note-intelligence-syudcrjlkpka3mewvvfnd2.streamlit.app/

---

## ⚠️ Disclaimer

This system is built for educational and portfolio purposes only.
It is not intended for real clinical decision making.
