import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
from pypdf import PdfReader
from groq import Groq

# 1. Page Configuration for a Professional Clinical Layout
st.set_page_config(
    page_title="Ovarian Cancer Diagnostic Support System",
    page_icon="🩺",
    layout="wide"
)

# 2. Initialize Groq Client (Picks up GROQ_API_KEY from environment)
try:
    groq_client = Groq(api_key="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")
except Exception as e:
    groq_client = None


# 3. Load Saved Model and Scaler Artifacts
@st.cache_resource
def load_diagnostic_artifacts():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


model, scaler, load_error = load_diagnostic_artifacts()

# UI Header & Instructions
st.title("🩺 Pathologist Portal: Instant PDF Report Analyzer:")
st.markdown("---")

if load_error:
    st.error(f"⚠️ Critical Error Loading Model Artifacts: {load_error}")
    st.info("Please ensure `model.pkl` and `scaler.pkl` exist in the exact same folder as `app.py`.")
    st.stop()

if groq_client is None:
    st.error("⚠️ Groq API Client could not be initialized. Please check your `GROQ_API_KEY` environment variable.")
    st.stop()

st.markdown(
    "Upload a patient's lab/diagnostic **PDF report**:")

# 4. File Uploader for PDF
uploaded_pdf = st.file_uploader("Upload Patient PDF Report", type=["pdf"])

if uploaded_pdf is not None:
    if st.button("Extract Data & Run Diagnostic Prediction", type="primary"):
        with st.spinner("Reading PDF locally and analyzing..."):
            try:
                # Step A: Extract text locally from the PDF (Instantaneous speed)
                reader = PdfReader(uploaded_pdf)
                pdf_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"

                # Step B: Define exact 48 training feature columns in correct order
                exact_features = [
                    "MPV", "BASO#", "PHOS", "GLU.", "K", "AST", "BASO%", "Mg", "Menopause", "CL",
                    "CEA", "EO#", "CA19-9", "ALB", "IBIL", "GGT", "MCH", "GLO", "ALT", "DBIL",
                    "Age", "RDW", "PDW", "CREA", "AFP", "HGB", "Na", "HE4", "LYM#", "CA125",
                    "BUN", "LYM%", "Ca", "AG", "MONO#", "PLT", "NEU", "EO%", "TP", "UA",
                    "RBC", "PCT", "CO2CP", "TBIL", "HCT", "MONO%", "MCV", "ALP"
                ]

                prompt = (
                    "You are an expert medical data extractor. Analyze the patient lab report text and extract values "
                    "for these exact 48 clinical features required by our diagnostic model. "
                    "You MUST use these exact string names as the JSON keys: "
                    f"{exact_features}\n\n"
                    "If a feature or test result is not mentioned or missing in the report, assign it a value of 0. "
                    "Return the data strictly as a flat JSON dictionary where keys are the feature names above and values are numbers."
                )

                # Step C: Call Groq API with JSON Mode enforced
                completion = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise medical data extraction assistant that outputs strict JSON."
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nReport Text:\n{pdf_text}"
                        }
                    ],
                    response_format={"type": "json_object"}
                )

                # Step D: Parse JSON, create DataFrame, and reorder to match training columns
                extracted_data = json.loads(completion.choices[0].message.content)
                input_df = pd.DataFrame([extracted_data])

                # Ensure columns strictly match training order
                for col in exact_features:
                    if col not in input_df.columns:
                        input_df[col] = 0.0  # Default missing columns to 0
                input_df = input_df[exact_features]

                st.subheader("📋 Extracted Clinical Features (Missing values defaulted to 0)")
                st.dataframe(input_df, use_container_width=True)

                # Step E: Scale features and run prediction model
                scaled_features = scaler.transform(input_df)
                raw_pred = model.predict(scaled_features)

                # Handle model output structure (Supports both Keras sigmoid arrays and scikit-learn predict_proba)
                if isinstance(raw_pred[0], (list, np.ndarray)):
                    pred_val_raw = raw_pred[0][0]
                else:
                    pred_val_raw = raw_pred[0]

                prediction_val = int(1 if pred_val_raw > 0.5 else 0)
                confidence = float(pred_val_raw) if prediction_val == 1 else float(1 - pred_val_raw)
                confidence_pct = confidence * 100

                # Step F: Display Results Dashboard
                st.markdown("---")
                st.subheader("🔬 AI Diagnostic Assessment Result")

                col1, col2 = st.columns(2)
                with col1:
                    if prediction_val == 1:
                        st.error("### Result: Malignant / High Risk")
                    else:
                        st.success("### Result: Benign / Low Risk")
                with col2:
                    st.metric(label="Model Confidence Score", value=f"{confidence_pct:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during extraction or evaluation: {e}")
