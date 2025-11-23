# streamlit_app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from math import pi

# Optional: Gemini (generative AI) - will be used only if API key exists
try:
    from google import generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# -------------------------
# App styling (dark/light)
# -------------------------
def local_css(theme="dark"):
    if theme == "dark":
        css = """
        <style>
        .stApp { background-color: #0e1117; color: #E6EDF3; }
        .stSidebar { background-color: #0b0f13; color: #E6EDF3; }
        .stButton>button { background-color:#1f6feb; color:white; border-radius:6px;}
        .stFileUploader>div{background-color:transparent;}
        .stMetric>div>div{color:#E6EDF3;}
        </style>
        """
    else:
        css = """
        <style>
        .stApp { background-color: #FFFFFF; color: #0b0f13; }
        .stSidebar { background-color: #F6F8FA; color: #0b0f13; }
        .stButton>button { background-color:#0d6efd; color:white; border-radius:6px;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Utility helpers
# -------------------------
@st.cache_data
def load_artifacts():
    artifacts = {}
    # Load model, scaler, label encoder safely
    artifacts['model'] = None
    artifacts['scaler'] = None
    artifacts['le_target'] = None

    if os.path.exists("ad_model.pkl"):
        try:
            artifacts['model'] = joblib.load("ad_model.pkl")
        except Exception as e:
            st.warning(f"Failed to load ad_model.pkl: {e}")

    if os.path.exists("scaler.pkl"):
        try:
            artifacts['scaler'] = joblib.load("scaler.pkl")
        except Exception as e:
            st.warning(f"Failed to load scaler.pkl: {e}")

    if os.path.exists("label_encoder.pkl"):
        try:
            artifacts['le_target'] = joblib.load("label_encoder.pkl")
        except Exception as e:
            st.warning(f"Failed to load label_encoder.pkl: {e}")

    return artifacts

def build_encoders_from_dataset(path="ads_synthetic.csv"):
    encoders = {}
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Fit encoders to match training
        for col in ["Gender", "City", "AdType"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                le.fit(df[col])
                encoders[col] = le
        return encoders
    except Exception:
        return None

# Fallback encoders (if dataset not present)
def fallback_encoders():
    encoders = {}
    encoders["Gender"] = LabelEncoder().fit(["Female", "Male"])  # alphabetical mapping
    encoders["City"] = LabelEncoder().fit(["Bangalore", "Chennai", "Hyderabad", "Kolkata", "Mumbai", "Delhi"])
    encoders["AdType"] = LabelEncoder().fit(["Carousel", "Image", "Shorts", "Video"])
    return encoders

def encode_row(row, encoders):
    # expects dict-like row
    r = row.copy()
    for c in ["Gender", "City", "AdType"]:
        if c in r:
            val = str(r[c])
            if c in encoders and val in encoders[c].classes_:
                r[c] = int(encoders[c].transform([val])[0])
            else:
                # unseen category handling: add to encoder classes by mapping to -1 -> then replace with most common (0)
                r[c] = 0
    return r

def plot_feature_importance(importances, feature_names):
    fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names)*0.4)))
    indices = np.argsort(importances)[::-1]
    sorted_feats = [feature_names[i] for i in indices]
    sorted_imps = importances[indices]
    ax.barh(sorted_feats[::-1], sorted_imps[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

def plot_radar(row_values, labels, title="Feature Radar"):
    # row_values: list of numeric values, labels: list of labels
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values = list(row_values)
    values += values[:1]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_title(title)
    st.pyplot(fig)

def call_gemini_prompt(features_dict, prediction_label):
    if not GEMINI_AVAILABLE:
        return "Gemini integration not available in this environment."
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY not set. Set env var to enable generative insights."
    try:
        genai.configure(api_key=api_key)
        prompt = f"""
You are an experienced ad performance analyst.

Input features: {features_dict}
Model prediction: {prediction_label}

Task:
1) Explain in 3 bullets why the model likely predicted {prediction_label}.
2) Provide 3 short actionable improvements (<=12 words each).
3) Suggest 2 alternative headlines (<=8 words each).
Keep tone concise and business-friendly.
"""
        resp = genai.generate_text(model="gemini-2.0-flash-lite", prompt=prompt, max_output_tokens=300, temperature=0.2)
        return resp.text
    except Exception as e:
        return f"Gemini call failed: {e}"

# -------------------------
# Main App
# -------------------------
st.set_page_config(page_title="AdPerf — Predictor + Insights", layout="wide", initial_sidebar_state="expanded")

# Sidebar controls
with st.sidebar:
    st.title("AdPerf Controls")
    theme = st.radio("Theme", options=["Dark", "Light"], index=0)
    local_css(theme.lower())
    st.markdown("---")
    st.markdown("🔽 Upload CSV to run batch predictions (expects same columns as training).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.markdown("⚙️ Model artifacts (loaded from working dir)")
    st.write("ad_model.pkl, scaler.pkl, label_encoder.pkl")
    st.markdown("---")
    st.markdown("ℹ️ Gemini insights (optional)")
    if GEMINI_AVAILABLE:
        st.write("Gemini SDK installed.")
    else:
        st.write("Gemini SDK not installed.")

# Load artifacts
artifacts = load_artifacts()
model = artifacts.get("model")
scaler = artifacts.get("scaler")
le_target = artifacts.get("le_target")

# Build encoders
encoders = build_encoders_from_dataset("ads_synthetic.csv")
if encoders is None:
    encoders = fallback_encoders()

# Page header
st.title("🚀 Ad Performance Predictor — Interactive Dashboard")
st.write("Upload a CSV or enter a single ad row. Model outputs class + probabilities and generative insights.")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1) Input Data")
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.success(f"Loaded {uploaded.name} — {df_in.shape[0]} rows")
            st.dataframe(df_in.head())
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            df_in = None
    else:
        st.info("No CSV uploaded — use manual input below.")
        # manual input form
        with st.form("manual_input"):
            Age = st.number_input("Age", min_value=16, max_value=100, value=28)
            Gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            City = st.selectbox("City", options=["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata"])
            AdType = st.selectbox("Ad Type", options=["Video","Image","Carousel","Shorts"])
            Budget = st.number_input("Budget (INR)", min_value=1000, max_value=2_000_000, value=50000, step=1000)
            DurationSec = st.number_input("Duration (sec)", min_value=1, max_value=300, value=30)
            Impressions = st.number_input("Impressions", min_value=0, max_value=10_000_000, value=200000, step=1000)
            CTR = st.number_input("CTR (%)", min_value=0.0, max_value=100.0, value=1.5, format="%.3f")
            Clicks = st.number_input("Clicks", min_value=0, max_value=1_000_000, value=int(Impressions*(CTR/100)))
            ConversionRate = st.number_input("Conversion Rate (%)", min_value=0.0, max_value=100.0, value=0.5, format="%.3f")
            Conversions = st.number_input("Conversions", min_value=0, max_value=1_000_000, value=int(Clicks*(ConversionRate/100)))
            submit_manual = st.form_submit_button("Use this row")
            if submit_manual:
                manual_row = {
                    "Age": Age,
                    "Gender": Gender,
                    "City": City,
                    "AdType": AdType,
                    "Budget": Budget,
                    "DurationSec": DurationSec,
                    "Impressions": Impressions,
                    "CTR": CTR,
                    "Clicks": Clicks,
                    "ConversionRate": ConversionRate,
                    "Conversions": Conversions
                }
                df_in = pd.DataFrame([manual_row])

with col2:
    st.subheader("2) Predict & Explain")
    if model is None:
        st.error("Model artifact not found: ad_model.pkl. Place it in the working directory.")
    else:
        st.write("Model loaded ✔")
    run_predict = st.button("Run Prediction")

# Prediction logic
if 'df_in' in locals() and df_in is not None and model is not None and scaler is not None and le_target is not None:
    if run_predict:
        # Ensure columns order same as training. We'll try to match automatically.
        expected_cols = ["Age","Gender","City","AdType","Budget","DurationSec","Impressions","CTR","Clicks","ConversionRate","Conversions"]
        missing = [c for c in expected_cols if c not in df_in.columns]
        if missing:
            st.error(f"Input is missing expected columns: {missing}")
        else:
            # Encode categorical columns
            df_proc = df_in.copy()
            for c in ["Gender","City","AdType"]:
                if c in df_proc.columns:
                    # use our encoders
                    try:
                        df_proc[c] = encoders[c].transform(df_proc[c].astype(str))
                    except Exception:
                        # fallback mapping
                        df_proc[c] = df_proc[c].astype(str).apply(lambda x: 0)
            # Convert to numeric matrix and scale
            X = df_proc[expected_cols].values.astype(float)
            try:
                Xs = scaler.transform(X)
            except Exception as e:
                st.warning(f"Scaler transform failed: {e}. Attempting to fit-transform as fallback.")
                Xs = scaler.fit_transform(X)

            preds = model.predict(Xs)
            probs = None
            try:
                probs = model.predict_proba(Xs)
            except Exception:
                probs = None

            # Map back to labels
            pred_labels = le_target.inverse_transform(preds.astype(int))
            st.metric("Predicted Performance", pred_labels[0])

            if probs is not None:
                # show probabilities for each class
                classes = le_target.classes_
                prob_series = {classes[i]: float(probs[0][i]) for i in range(len(classes))}
                st.subheader("Prediction Probabilities")
                st.write(pd.DataFrame([prob_series]).T.rename(columns={0:"probability"}))

            # Feature importance (if available)
            st.subheader("Feature Importance")
            try:
                if hasattr(model, "feature_importances_"):
                    feature_names = expected_cols
                    imps = model.feature_importances_
                    plot_feature_importance(imps, feature_names)
                else:
                    st.write("Feature importance not available for this model object.")
            except Exception as e:
                st.write("Couldn't compute feature importance:", e)

            # Radar for numeric features: choose a subset
            st.subheader("Feature Radar (selected numeric features)")
            radar_features = ["Budget","Impressions","CTR","Clicks","ConversionRate","Conversions"]
            radar_vals = [float(df_in.iloc[0][f]) if f in df_in.columns else 0.0 for f in radar_features]
            # normalize radar values for display (0-1)
            norm_vals = []
            for i, f in enumerate(radar_features):
                v = radar_vals[i]
                # simple normalization heuristics
                if f == "Budget":
                    norm_vals.append(min(v/200000.0, 1.0))
                elif f == "Impressions":
                    norm_vals.append(min(v/1_000_000.0, 1.0))
                elif f == "CTR":
                    norm_vals.append(min(v/10.0, 1.0))
                elif f == "Clicks":
                    norm_vals.append(min(v/50000.0, 1.0))
                elif f == "ConversionRate":
                    norm_vals.append(min(v/5.0, 1.0))
                elif f == "Conversions":
                    norm_vals.append(min(v/5000.0, 1.0))
                else:
                    norm_vals.append(0.0)
            plot_radar(norm_vals, radar_features, title="Normalized Numeric Feature Radar")

            # Gemini insights (optional)
            st.subheader("Generative Insights (Gemini)")
            if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
                with st.spinner("Calling Gemini..."):
                    insight = call_gemini_prompt(df_in.iloc[0].to_dict(), pred_labels[0])
                    st.markdown(insight.replace("\n", "  \n"))
            else:
                st.info("Gemini insights not available. Set GEMINI_API_KEY environment variable and install google-generativeai SDK to enable.")

elif 'df_in' in locals() and df_in is not None:
    st.warning("Model artifacts missing (ad_model.pkl, scaler.pkl, label_encoder.pkl). You can still view the input data above.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ for ad campaign prototyping — Streamlit app (dark/light mode).")

