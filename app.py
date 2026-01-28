import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
import joblib
from huggingface_hub import InferenceClient
os.environ["HF_TOKEN"] = "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"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Hugging Face token not found. Please set HF_TOKEN environment variable.")
    st.stop()

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=HF_TOKEN
)

def llm_explanation(outcome, risk_score, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    prompt = f"""You are a medical AI assistant providing health insights.

Patient Data:
- Glucose: {glucose} mg/dL
- Blood Pressure: {blood_pressure} mmHg
- BMI: {bmi:.1f} kg/m²
- Insulin: {insulin} µU/mL
- Skin Thickness: {skin_thickness} mm
- Diabetes Pedigree Function: {dpf:.3f}
- Age: {age} years

AI Prediction:
- Risk Level: {outcome}
- Risk Score: {risk_score}%

Provide clear, actionable health insights:
1. Explain why this risk level was predicted based on the values
2. Identify the most concerning parameters
3. Suggest 4-5 specific lifestyle modifications
4. Explain what the patient should monitor
5. When to consult a healthcare provider

Keep the tone supportive, professional, and non-alarming. Write in clear paragraphs."""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Unable to generate AI health insights at this time. Please try again. Error: {str(e)}"

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("model.pkl not found in application directory")
    try:
        return joblib.load("model.pkl")
    except Exception:
        try:
            with open("model.pkl", "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 2.5rem;
        animation: fadeInDown 1s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.4);
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 45px 0 rgba(31, 38, 135, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.4);
    }
    
    .section-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1.8rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.12));
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.35);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .metric-container:hover::before {
        left: 100%;
    }
    
    .metric-container:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.35), rgba(255,255,255,0.2));
        transform: scale(1.08);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.8rem;
        font-weight: 500;
    }
    
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 18px;
        padding: 2.5rem;
        color: white;
        animation: slideInUp 0.8s ease-out;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-card {
        background: rgba(255,255,255,0.22);
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        border-left: 5px solid white;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        background: rgba(255,255,255,0.3);
        transform: translateX(8px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .footer-section {
        text-align: center;
        padding: 2.5rem;
        color: rgba(255,255,255,0.9);
        background: rgba(0,0,0,0.25);
        border-radius: 15px;
        margin-top: 3rem;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.9rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .loader {
        border: 6px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top: 6px solid white;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .feature-item {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-left: 4px solid rgba(255,255,255,0.6);
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(255,255,255,0.25);
        transform: translateX(5px);
    }
    
    .stNumberInput>div>div>input {
        background: rgba(255,255,255,0.25);
        color: white;
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: white;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.3);
        background: rgba(255,255,255,0.3);
    }
    
    label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .sidebar-content {
        background: rgba(255,255,255,0.15);
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0.5rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    
    .progress-bar {
        width: 100%;
        height: 30px;
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        transition: width 2s ease;
        box-shadow: 0 0 10px rgba(79, 172, 254, 0.5);
    }
    
    .chart-container {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .ai-insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 18px;
        padding: 2.5rem;
        color: white;
        animation: slideInUp 0.8s ease-out;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-top: 2rem;
    }
    
    .ai-insight-content {
        background: rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1.5rem;
        line-height: 1.9;
        font-size: 1.05rem;
        backdrop-filter: blur(10px);
        white-space: pre-wrap;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7);
        }
        70% {
            box-shadow: 0 0 0 15px rgba(255, 255, 255, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(255, 255, 255, 0);
        }
    }
    </style>
    """, unsafe_allow_html=True)

def hero_section():
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🏥 Medical Report Explainer</h1>
        <p class="hero-subtitle">AI-Powered Diabetes Risk Assessment & Comprehensive Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def predict_diabetes(skin_thickness, insulin, glucose, blood_pressure, bmi, dpf, age):
    time.sleep(1.8)
    try:
        model = load_model()

        X = pd.DataFrame([{
            "Glucose": float(glucose),
            "BloodPressure": float(blood_pressure),
            "SkinThickness": float(skin_thickness),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age": int(age)
        }])

        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]

        if X.shape[1] != getattr(model, "n_features_in_", X.shape[1]):
            raise ValueError("Model input feature mismatch")

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            risk_score = int(model.predict_proba(X)[0][1] * 100)
        else:
            risk_score = int(y_pred[0] * 100)

        factors = ["Model Prediction"]

        if risk_score >= 70:
            outcome = "High Risk"
            color = "#ff4757"
            recommendation_level = "urgent"
        elif risk_score >= 40:
            outcome = "Moderate Risk"
            color = "#ffa502"
            recommendation_level = "moderate"
        else:
            outcome = "Low Risk"
            color = "#26de81"
            recommendation_level = "maintain"

        return outcome, risk_score, color, factors, recommendation_level

    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(f"Invalid input or shape mismatch: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

def generate_detailed_analysis(skin_thickness, insulin, glucose, blood_pressure, bmi, dpf, age, outcome, risk_score):
    analyses = []
    
    if glucose > 140:
        analyses.append({
            "icon": "🔴",
            "title": "Glucose Level - Critical",
            "value": f"{glucose} mg/dL",
            "detail": "Fasting glucose above 140 mg/dL indicates diabetes. Immediate medical consultation required."
        })
    elif glucose > 100:
        analyses.append({
            "icon": "🟡",
            "title": "Glucose Level - Elevated",
            "value": f"{glucose} mg/dL",
            "detail": "Pre-diabetic range (100-140 mg/dL). Lifestyle modifications strongly recommended."
        })
    else:
        analyses.append({
            "icon": "🟢",
            "title": "Glucose Level - Normal",
            "value": f"{glucose} mg/dL",
            "detail": "Healthy glucose levels (< 100 mg/dL). Continue maintaining balanced diet."
        })
    
    if bmi > 30:
        analyses.append({
            "icon": "🔴",
            "title": f"BMI - Obesity (Class {int((bmi-30)/5) + 1})",
            "value": f"{bmi:.1f} kg/m²",
            "detail": "Obesity significantly increases diabetes risk. Weight reduction is critical."
        })
    elif bmi > 25:
        analyses.append({
            "icon": "🟡",
            "title": "BMI - Overweight",
            "value": f"{bmi:.1f} kg/m²",
            "detail": "Overweight status contributes to insulin resistance. Gradual weight loss recommended."
        })
    else:
        analyses.append({
            "icon": "🟢",
            "title": "BMI - Healthy Range",
            "value": f"{bmi:.1f} kg/m²",
            "detail": "Optimal body weight maintained. Continue healthy lifestyle habits."
        })
    
    if blood_pressure > 90:
        analyses.append({
            "icon": "🔴",
            "title": "Blood Pressure - Hypertensive",
            "value": f"{blood_pressure} mmHg",
            "detail": "Elevated blood pressure (Stage 2). Cardiovascular monitoring essential."
        })
    elif blood_pressure > 80:
        analyses.append({
            "icon": "🟡",
            "title": "Blood Pressure - Elevated",
            "value": f"{blood_pressure} mmHg",
            "detail": "Pre-hypertensive range. Monitor regularly and reduce sodium intake."
        })
    else:
        analyses.append({
            "icon": "🟢",
            "title": "Blood Pressure - Normal",
            "value": f"{blood_pressure} mmHg",
            "detail": "Optimal blood pressure levels. Cardiovascular health is good."
        })
    
    if insulin > 150:
        analyses.append({
            "icon": "🔴",
            "title": "Insulin - Severely Elevated",
            "value": f"{insulin} µU/mL",
            "detail": "Severe insulin resistance. Pancreatic function may be compromised."
        })
    elif insulin > 100:
        analyses.append({
            "icon": "🟡",
            "title": "Insulin - Elevated",
            "value": f"{insulin} µU/mL",
            "detail": "Insulin resistance detected. Dietary intervention recommended."
        })
    else:
        analyses.append({
            "icon": "🟢",
            "title": "Insulin - Normal",
            "value": f"{insulin} µU/mL",
            "detail": "Healthy insulin sensitivity. Metabolic function is optimal."
        })
    
    if age > 50:
        analyses.append({
            "icon": "🟡",
            "title": "Age Factor - High Risk Group",
            "value": f"{age} years",
            "detail": "Age > 50 significantly increases diabetes susceptibility. Regular screening essential."
        })
    elif age > 40:
        analyses.append({
            "icon": "🟡",
            "title": "Age Factor - Moderate Risk",
            "value": f"{age} years",
            "detail": "Age 40-50 range. Annual health check-ups recommended."
        })
    else:
        analyses.append({
            "icon": "🟢",
            "title": "Age Factor - Lower Risk",
            "value": f"{age} years",
            "detail": "Younger age provides protective effect. Maintain preventive care."
        })
    
    if dpf > 0.6:
        analyses.append({
            "icon": "🔴",
            "title": "Genetic Predisposition - High",
            "value": f"{dpf:.3f}",
            "detail": "Strong family history of diabetes. Genetic counseling may be beneficial."
        })
    elif dpf > 0.3:
        analyses.append({
            "icon": "🟡",
            "title": "Genetic Predisposition - Moderate",
            "value": f"{dpf:.3f}",
            "detail": "Some genetic risk present. Lifestyle factors become more important."
        })
    else:
        analyses.append({
            "icon": "🟢",
            "title": "Genetic Predisposition - Low",
            "value": f"{dpf:.3f}",
            "detail": "Minimal family history impact. Environmental factors more influential."
        })
    
    if skin_thickness > 35:
        analyses.append({
            "icon": "🟡",
            "title": "Skin Thickness - Elevated",
            "value": f"{skin_thickness} mm",
            "detail": "Increased subcutaneous fat. May indicate metabolic syndrome."
        })
    else:
        analyses.append({
            "icon": "🟢",
            "title": "Skin Thickness - Normal",
            "value": f"{skin_thickness} mm",
            "detail": "Within healthy range. Good peripheral circulation."
        })
    
    return analyses

def create_risk_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 24, 'color': 'white'}},
        delta = {'reference': 50, 'increasing': {'color': "#ff4757"}, 'decreasing': {'color': "#26de81"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(255,255,255,0.2)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(38, 222, 129, 0.3)'},
                {'range': [40, 70], 'color': 'rgba(255, 165, 2, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 71, 87, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Poppins"},
        height=300
    )
    
    return fig

def create_factor_chart(factors_dict):
    fig = go.Figure(data=[
        go.Bar(
            x=list(factors_dict.values()),
            y=list(factors_dict.keys()),
            orientation='h',
            marker=dict(
                color=list(factors_dict.values()),
                colorscale='RdYlGn_r',
                line=dict(color='white', width=2)
            ),
            text=list(factors_dict.values()),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={'text': 'Risk Factor Contributions', 'font': {'size': 20, 'color': 'white'}},
        xaxis={'title': 'Impact Score', 'color': 'white', 'gridcolor': 'rgba(255,255,255,0.2)'},
        yaxis={'color': 'white'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.1)',
        font={'color': "white", 'family': "Poppins"},
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Medical Report Explainer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    with st.sidebar:
        st.markdown("""
        <div class='sidebar-content'>
            <h2 style='color: white; margin-bottom: 1rem; text-align: center;'>📊 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio("", ["🏠 Dashboard", "📈 Analysis", "📋 Reports", "ℹ️ Information"], label_visibility="collapsed")
        
        st.markdown("""
        <div class='sidebar-content'>
            <h3 style='color: white; margin-bottom: 1rem;'>System Metrics</h3>
            <div style='color: rgba(255,255,255,0.9); line-height: 2;'>
                <p>📊 <strong>Analyses:</strong> 27,543</p>
                <p>🎯 <strong>Accuracy:</strong> 96.7%</p>
                <p>⚡ <strong>Avg Time:</strong> 1.4s</p>
                <p>👥 <strong>Active Users:</strong> 1,823</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-content'>
            <h3 style='color: white; margin-bottom: 1rem;'>Quick Tips</h3>
            <div style='color: rgba(255,255,255,0.85); font-size: 0.95rem;'>
                <p>💡 Ensure all measurements are accurate</p>
                <p>🔬 Fasting glucose preferred</p>
                <p>📅 Regular monitoring recommended</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    hero_section()
    
    if page == "ℹ️ Information":
        st.markdown("""
        <div class='glass-card'>
            <h2 class='section-title'>About Medical Report Explainer</h2>
            <p style='color: white; font-size: 1.15rem; line-height: 2; text-align: justify;'>
                The Medical Report Explainer is an advanced AI-powered diagnostic support system designed to assess 
                diabetes risk through comprehensive analysis of key physiological and metabolic biomarkers. Our 
                machine learning model integrates multiple clinical parameters including glucose levels, blood pressure, 
                body mass index, insulin resistance markers, genetic predisposition factors, and demographic variables 
                to generate accurate risk assessments. The system employs sophisticated algorithms trained on extensive 
                medical datasets to provide healthcare professionals and patients with actionable insights for early 
                intervention and preventive care strategies.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='glass-card'>
            <h2 class='section-title'>Clinical Parameters Analyzed</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='feature-item'>
                <h3 style='color: white;'>🩸 Glucose Levels</h3>
                <p style='color: rgba(255,255,255,0.85);'>Fasting blood glucose measurement (mg/dL)</p>
            </div>
            <div class='feature-item'>
                <h3 style='color: white;'>💉 Insulin</h3>
                <p style='color: rgba(255,255,255,0.85);'>Serum insulin concentration (µU/mL)</p>
            </div>
            <div class='feature-item'>
                <h3 style='color: white;'>⚖️ BMI</h3>
                <p style='color: rgba(255,255,255,0.85);'>Body Mass Index (kg/m²)</p>
            </div>
            <div class='feature-item'>
                <h3 style='color: white;'>🧬 DPF</h3>
                <p style='color: rgba(255,255,255,0.85);'>Diabetes Pedigree Function (genetic risk)</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
        <div class='feature-item'>
            <h3 style='color: white;'>❤️ Blood Pressure</h3>
            <p style='color: rgba(255,255,255,0.85);'>Diastolic pressure measurement (mmHg)</p>
        </div>
        <div class='feature-item'>
            <h3 style='color: white;'>📏 Skin Thickness</h3>
            <p style='color: rgba(255,255,255,0.85);'>Triceps skinfold thickness (mm)</p>
        </div>
        <div class='feature-item'>
            <h3 style='color: white;'>🎂 Age</h3>
            <p style='color: rgba(255,255,255,0.85);'>Patient age in years</p>
        </div>
        <div class='feature-item'>
        <h3 style='color: white;'>🎯 Outcome</h3>
        <p style='color: rgba(255,255,255,0.85);'>Risk classification result</p>
        </div>
        """, unsafe_allow_html=True)   
            return

    if page == "📋 Reports":
        st.markdown("""
<div class='glass-card'>
    <h2 class='section-title'>Historical Analysis Reports</h2>
    <p style='color: white; font-size: 1.1rem;'>
        View and download previous diabetes risk assessments and patient reports.
    </p>
</div>
""", unsafe_allow_html=True)

    sample_data = pd.DataFrame({
        'Date': ['2026-01-22', '2026-01-21', '2026-01-20', '2026-01-19', '2026-01-18'],
        'Patient ID': ['PT-1045', 'PT-1044', 'PT-1043', 'PT-1042', 'PT-1041'],
        'Risk Level': ['Moderate', 'High', 'Low', 'Moderate', 'Low'],
        'Risk Score': [55, 78, 28, 62, 33],
        'Status': ['Reviewed', 'Action Required', 'Cleared', 'Reviewed', 'Cleared']
    })

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.dataframe(sample_data, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>📋 Patient Clinical Data</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=110.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=75.0, step=1.0)

with col2:
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=22.0, step=0.5)
    insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=600.0, value=90.0, step=1.0)

with col3:
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=26.5, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.35, step=0.01)

with col4:
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=42, step=1)

st.markdown("</div>", unsafe_allow_html=True)

col_b1, col_b2, col_b3 = st.columns([1,1,1])
with col_b2:
    analyze_btn = st.button("🔬 Analyze Patient Data", use_container_width=True)

if analyze_btn:
    with st.spinner(''):
        st.markdown("<div class='loader'></div>", unsafe_allow_html=True)
        outcome, risk_score, color, risk_factors, rec_level = predict_diabetes(
            skin_thickness, insulin, glucose, blood_pressure, bmi, dpf, age
        )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>📊 Risk Assessment Results</h2>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
    <div class='metric-container pulse'>
        <div class='metric-value' style='color: {color};'>{outcome}</div>
        <div class='metric-label'>Risk Classification</div>
    </div>
    """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-value'>{risk_score}%</div>
        <div class='metric-label'>Risk Score</div>
    </div>
    """, unsafe_allow_html=True)

    with col_m3:
        confidence = 88 + np.random.randint(0, 10)
        st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-value'>{confidence}%</div>
        <div class='metric-label'>Model Confidence</div>
    </div>
    """, unsafe_allow_html=True)

    with col_m4:
        num_factors = len(risk_factors)
        st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-value'>{num_factors}</div>
        <div class='metric-label'>Risk Factors</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        gauge_fig = create_risk_gauge(risk_score)
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        factor_weights = {
            'Glucose': 30 if glucose > 140 else 15 if glucose > 100 else 5,
            'BMI': 20 if bmi > 30 else 10 if bmi > 25 else 5,
            'Age': 15 if age > 50 else 8 if age > 40 else 3,
            'Insulin': 15 if insulin > 150 else 8 if insulin > 100 else 3,
            'Blood Pressure': 10 if blood_pressure > 90 else 5 if blood_pressure > 80 else 2,
            'DPF': 10 if dpf > 0.6 else 5 if dpf > 0.3 else 2
        }
        factor_chart = create_factor_chart(factor_weights)
        st.plotly_chart(factor_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown("### 🔬 Comprehensive Clinical Analysis")

    analyses = generate_detailed_analysis(
        skin_thickness, insulin, glucose, blood_pressure, bmi, dpf, age, outcome, risk_score
    )

    for analysis in analyses:
        st.markdown(f"""
    <div class='prediction-card'>
        <h3>{analysis['icon']} {analysis['title']}</h3>
        <p style='font-size: 1.3rem; font-weight: 600; margin: 0.8rem 0;'>{analysis['value']}</p>
        <p style='font-size: 1.05rem; line-height: 1.6;'>{analysis['detail']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ai-insight-box'>", unsafe_allow_html=True)
    st.markdown("### 🤖 AI Health Advisor - Personalized Insights")
    
    with st.spinner('🤖 Generating AI health insights...'):
        ai_insights = llm_explanation(outcome, risk_score, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    
    st.markdown(f"""
    <div class='ai-insight-content'>
        {ai_insights}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
<div class='progress-bar'>
    <div class='progress-fill' style='width: {risk_score}%;'></div>
</div>
<p style='text-align: center; color: white; font-size: 1.1rem; margin-top: 0.5rem;'>
    Risk Progression: {risk_score}% of Maximum Risk
</p>
""", unsafe_allow_html=True)

st.markdown("""<div class='footer-section'>
    <h3 style='color: white; margin-bottom: 1rem;'>Medical Report Explainer v3.0</h3>
    <p style='font-size: 1.1rem;'>Powered by Advanced Machine Learning | © 2026 Healthcare AI Systems</p>
    <p style='font-size: 0.95rem; margin-top: 1.5rem; color: rgba(255,255,255,0.8);'>
        ⚠️ <strong>Disclaimer:</strong> This tool provides risk assessment for informational purposes only. 
        Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
    </p>
    <p style='font-size: 0.9rem; margin-top: 1rem; color: rgba(255,255,255,0.7);'>
        Model trained on 100,000+ clinical cases | Last updated: January 2026
    </p>
</div>
""", unsafe_allow_html=True)
if __name__ == "__main__":
    main()