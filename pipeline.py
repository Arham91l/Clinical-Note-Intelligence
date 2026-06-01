import os
import torch
import joblib
import numpy as np
from transformers import BartForConditionalGeneration, AutoTokenizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)

# ── Model 1: Summarization ───────────────────────────────────
print("Loading summarization model...")
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained(model_name)
bart_model = bart_model.to("cuda" if torch.cuda.is_available() else "cpu")
print("Summarization model loaded ✅")

# ── Model 2: Keyword Extraction ──────────────────────────────
print("Loading keyword extraction model...")
kw_encoder = SentenceTransformer(
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)
kw_model = KeyBERT(model=kw_encoder)
print("Keyword model loaded ✅")

# ── Model 3: Risk Prediction ─────────────────────────────────
# ── Model 3: Risk Prediction ─────────────────────────────────
print("Loading risk prediction model...")
risk_model   = joblib.load(os.path.join(BASE_DIR, "models/risk_model.pkl"))
risk_encoder = joblib.load(os.path.join(BASE_DIR, "models/risk_encoder.pkl"))
ohe          = joblib.load(os.path.join(BASE_DIR, "models/ohe_encoder.pkl"))

# Load directly from HuggingFace instead of pkl
sent_encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("Risk model loaded ✅")


def chunk_text(text, max_tokens=900, overlap=50):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    start  = 0
    while start < len(tokens):
        end   = min(start + max_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


def summarize_text(text, min_len=60, max_len=180):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(bart_model.device)

    with torch.no_grad():
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize(text, min_len=60, max_len=180):
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) <= 1024:
        return summarize_text(text, min_len, max_len)

    # Long text: chunk → summarize each → final pass
    chunks          = chunk_text(text)
    chunk_summaries = [summarize_text(c, 40, 120) for c in chunks]
    combined        = " ".join(chunk_summaries)
    return summarize_text(combined, min_len, max_len)


def extract_keywords(summary, top_n=10):
    if not summary or len(summary.strip()) == 0:
        return ""
    keywords = kw_model.extract_keywords(
        summary,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_mmr=True,
        diversity=0.5,
        top_n=top_n
    )
    return ", ".join([kw[0] for kw in keywords])


def predict_risk(keywords, specialty):
    keyword_emb   = sent_encoder.encode([keywords])
    specialty_ohe = ohe.transform([[specialty]])
    X             = np.hstack([keyword_emb, specialty_ohe])
    pred          = risk_model.predict(X)
    proba         = risk_model.predict_proba(X)[0]
    label         = risk_encoder.inverse_transform(pred)[0]
    confidence    = {
        cls: round(float(prob), 3)
        for cls, prob in zip(risk_encoder.classes_, proba)
    }
    return label, confidence


def run_pipeline(transcription, description, specialty):
    combined   = description + " " + transcription
    summary    = summarize(combined)
    keywords   = extract_keywords(summary)
    risk_label, confidence = predict_risk(keywords, specialty)
    return {
        "summary"    : summary,
        "keywords"   : keywords,
        "risk_label" : risk_label,
        "confidence" : confidence
    }
