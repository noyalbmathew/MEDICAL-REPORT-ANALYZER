import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import os, pickle, joblib
from huggingface_hub import InferenceClient

# --------------------------------------------------
# 🔑 LLM SETUP (SAFE – NO secrets.toml REQUIRED)
# --------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Hugging Face token not found. Please set HF_TOKEN environment variable.")
    st.stop()

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=HF_TOKEN
)

def llm_explanation(outcome, risk_score, inputs):
    prompt = f"""
You are a medical AI assistant.

Patient details:
- Glucose: {inputs['glucose']} mg/dL
- Blood Pressure: {inputs['blood_pressure']} mmHg
- BMI: {inputs['bmi']}
- Insulin: {inputs['insulin']} µU/mL
- Skin Thickness: {inputs['skin_thickness']} mm
- Diabetes Pedigree Function: {inputs['dpf']}
- Age: {inputs['age']}

Prediction:
- Risk Level: {outcome}
- Risk Score: {risk_score}%

Explain clearly:
1. Why this risk level was predicted
2. What it means in simple terms
3. Practical next steps

Keep it supportive and non-alarming.
"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )

    return response.choices[0].message.content


# --------------------------------------------------
# 🧠 ML MODEL LOADING
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("model.pkl not found")
    try:
        return joblib.load("model.pkl")
    except:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)


# --------------------------------------------------
# 🔮 PREDICTION LOGIC
# --------------------------------------------------
def predict_diabetes(skin_thickness, insulin, glucose, blood_pressure, bmi, dpf, age):
    model = load_model()

    X = pd.DataFrame([{
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    if hasattr(model, "predict_proba"):
        risk_score = int(model.predict_proba(X)[0][1] * 100)
    else:
        risk_score = int(model.predict(X)[0] * 100)

    if risk_score >= 70:
        outcome = "High Risk"
        color = "#ff4757"
    elif risk_score >= 40:
        outcome = "Moderate Risk"
        color = "#ffa502"
    else:
        outcome = "Low Risk"
        color = "#26de81"

    return outcome, risk_score, color


# --------------------------------------------------
# 🌐 STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="Medical Report Explainer (ML + LLM)",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Report Explainer")
st.caption("Diabetes Risk Prediction with Machine Learning + LLM Explanation")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, 110.0)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 75.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0.0, 100.0, 22.0)

with col2:
    insulin = st.number_input("Insulin (µU/mL)", 0.0, 600.0, 90.0)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 26.5)

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.35)
    age = st.number_input("Age (years)", 18, 120, 42)

st.divider()

if st.button("🔬 Analyze & Explain", use_container_width=True):

    with st.spinner("Running ML prediction..."):
        outcome, risk_score, color = predict_diabetes(
            skin_thickness, insulin, glucose,
            blood_pressure, bmi, dpf, age
        )

    st.subheader("📊 Risk Assessment")
    st.metric("Risk Level", outcome)
    st.metric("Risk Score", f"{risk_score}%")

    inputs = {
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "dpf": dpf,
        "age": age
    }

    with st.spinner("Generating AI medical explanation..."):
        explanation = llm_explanation(outcome, risk_score, inputs)

    st.subheader("🧠 AI Medical Explanation")
    st.markdown(
        f"""
        <div style="background:#111; padding:20px; border-radius:10px; color:white;">
        {explanation}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("⚠️ This AI explanation is for informational purposes only and not a medical diagnosis.")

st.divider()
st.caption("Powered by ML + Llama-3.1-8B-Instruct | © 2026")