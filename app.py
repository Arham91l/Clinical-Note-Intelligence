import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from pipeline import run_pipeline

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Note Intelligence System",
    page_icon="🏥",
    layout="wide"
)
# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #1a237e, #0d47a1);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Text areas and inputs */
    .stTextArea textarea {
        background-color: #1e2130;
        color: #ffffff;
        border: 1px solid #3d4466;
    }

    /* Selectbox */
    .stSelectbox div {
        background-color: #1e2130;
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
    }

    /* Risk cards */
    .risk-high {
        background-color: #2d1b1b;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
        color: #ff8a80;
    }
    .risk-medium {
        background-color: #2d2416;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
        color: #ffcc80;
    }
    .risk-low {
        background-color: #1b2d1e;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        color: #a5d6a7;
    }

    /* Keyword badges */
    .keyword-badge {
        display: inline-block;
        background-color: #1a237e;
        color: #90caf9;
        padding: 4px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 13px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 10px;
    }

    /* Divider */
    hr {
        border-color: #3d4466;
    }

    /* Info boxes */
    .stInfo {
        background-color: #1a237e;
        color: #ffffff;
    }

    /* Button */
    .stButton button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #1557b0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Clinical Note Intelligence System</h1>
    <p>Automated Summarization · Keyword Extraction · Risk Prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("⚙️ About")
st.sidebar.markdown("### 3-Model Cascaded Pipeline")
st.sidebar.info(
    "**Model 1** — Abstractive Summarization\n\n"
    "Condenses clinical notes into a concise patient summary.\n\n"
    "**Model 2** — Keyword Extraction\n\n"
    "Extracts key medical terms using BioBERT + cosine similarity.\n\n"
    "**Model 3** — Risk Prediction\n\n"
    "Predicts patient risk level using XGBoost with 92% accuracy."
)

st.sidebar.markdown("### Dataset")
st.sidebar.success("MTSamples — 4,921 real clinical notes")

st.sidebar.markdown("### Model Performance")
st.sidebar.markdown("""
| Model | Metric | Score |
|---|---|---|
| Summarization | ROUGE-2 | 0.29 |
| Keywords | Cosine Sim | 0.62 |
| Risk | Accuracy | 0.92 |
""")

# ── Load Specialties ─────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data/mtsamples_final.csv")
df         = pd.read_csv(DATA_PATH)
specialties = sorted(df["medical_specialty"].dropna().str.strip().unique().tolist())

# ── Input Section ────────────────────────────────────────────
st.subheader("📋 Patient Clinical Note Input")

col1, col2 = st.columns([2, 1])
with col1:
    description = st.text_area(
        "Brief Description (optional)",
        placeholder="e.g. Left heart catheterization, selective coronary angiography...",
        height=100
    )
with col2:
    specialty = st.selectbox(
        "Medical Specialty",
        options=specialties,
        index=0
    )

transcription = st.text_area(
    "Full Clinical Note / Transcription",
    placeholder="Paste the complete clinical note here...",
    height=280
)

# ── Sample Note Button ───────────────────────────────────────
if st.button("📄 Load Sample Note", use_container_width=False):
    sample = df[df["medical_specialty"].str.strip() == "Cardiovascular / Pulmonary"].iloc[0]
    st.session_state["sample_transcription"] = sample["transcription"]
    st.session_state["sample_description"]   = sample["description"]
    st.rerun()

if "sample_transcription" in st.session_state:
    transcription = st.session_state.pop("sample_transcription")
    description   = st.session_state.pop("sample_description")

# ── Analyze Button ───────────────────────────────────────────
analyze = st.button(
    "🔍 Analyze Clinical Note",
    type="primary",
    use_container_width=True
)

if analyze:
    if not transcription.strip():
        st.error("⚠️ Please enter a clinical note to analyze.")
    else:
        with st.spinner("⏳ Running pipeline... Model 1 → Model 2 → Model 3"):
            result = run_pipeline(transcription, description, specialty)

        st.divider()
        st.subheader("📊 Analysis Results")

        # ── Row 1: Summary + Keywords ─────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📝 Patient Summary")
            st.info(result["summary"])

        with col2:
            st.markdown("### 🔑 Extracted Keywords")
            keywords_html = "".join([
                f'<span class="keyword-badge">{kw.strip()}</span>'
                for kw in result["keywords"].split(",")
            ])
            st.markdown(keywords_html, unsafe_allow_html=True)

        st.divider()

        # ── Row 2: Risk Assessment ────────────────────────────
        st.markdown("### 🚨 Risk Assessment")

        risk       = result["risk_label"]
        confidence = result["confidence"]

        if risk == "High":
            st.markdown(
                f'<div class="risk-high"><h3>⚠️ Risk Level: HIGH</h3>'
                f'<p>This patient requires immediate clinical attention.</p></div>',
                unsafe_allow_html=True
            )
        elif risk == "Medium":
            st.markdown(
                f'<div class="risk-medium"><h3>🟡 Risk Level: MEDIUM</h3>'
                f'<p>This patient requires monitoring and follow-up.</p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="risk-low"><h3>✅ Risk Level: LOW</h3>'
                f'<p>This patient appears stable. Routine care recommended.</p></div>',
                unsafe_allow_html=True
            )

        st.markdown("#### Confidence Scores")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "🔴 High Risk",
                f"{confidence.get('High', 0)*100:.1f}%"
            )
        with c2:
            st.metric(
                "🟡 Medium Risk",
                f"{confidence.get('Medium', 0)*100:.1f}%"
            )
        with c3:
            st.metric(
                "🟢 Low Risk",
                f"{confidence.get('Low', 0)*100:.1f}%"
            )

        st.divider()

        # ── Row 3: Raw Output ─────────────────────────────────
        with st.expander("🔎 View Raw Pipeline Output"):
            st.json(result)
