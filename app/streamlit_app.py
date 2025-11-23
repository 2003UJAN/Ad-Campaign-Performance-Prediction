import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Ad Campaign Performance Prediction",
    page_icon="📊",
    layout="centered"
)

# -------------------------
# CUSTOM LIGHT / DARK THEMING
# -------------------------
MODE = st.sidebar.radio("🌗 Theme Mode", ["Light", "Dark"])

if MODE == "Dark":
    st.markdown("""
        <style>
            body, .stApp { background-color: #0e1117; color: #f0f0f0; }
            .stTextInput>div>div>input { background-color: #1c1f26; color: #fff; }
            .stSelectbox>div>div { background-color: #1c1f26; color: #fff; }
            .stNumberInput>div>div>input { background-color: #1c1f26; color: #fff; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp { background-color: #f8f9fa; color: #000; }
        </style>
    """, unsafe_allow_html=True)


# -------------------------
# ARTIFACT LOADER
# -------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {"model": None, "scaler": None, "le_target": None}

    try:
        if os.path.exists("ad_model.pkl"):
            artifacts["model"] = joblib.load("ad_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

    try:
        if os.path.exists("scaler.pkl"):
            artifacts["scaler"] = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load scaler: {e}")

    try:
        if os.path.exists("label_encoder.pkl"):
            artifacts["le_target"] = joblib.load("label_encoder.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load label encoder: {e}")

    return artifacts


artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
le_target = artifacts["le_target"]

# -------------------------
# VALIDATION MESSAGE
# -------------------------
if not all([model, scaler, le_target]):
    st.warning("⚠ Model files missing! Ensure **ad_model.pkl**, **scaler.pkl**, and **label_encoder.pkl** are in the same directory.")
else:
    st.success("✅ Model Loaded Successfully!")


# -------------------------
# PAGE HEADER
# -------------------------
st.title("📊 Ad Campaign Performance Predictor")
st.write("Enter campaign details below to predict performance (High / Medium / Low).")


# -------------------------
# USER INPUT FORM
# -------------------------
with st.form("prediction_form"):
    st.subheader("📌 Campaign Inputs")

    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    budget = st.number_input("Ad Budget ($)", min_value=100, max_value=100000, value=5000)
    duration = st.number_input("Campaign Duration (days)", min_value=1, max_value=60, value=10)
    platform = st.selectbox("Platform", ["Instagram", "YouTube", "Facebook", "LinkedIn"])
    creatives = st.number_input("Number of Creatives", min_value=1, max_value=20, value=4)
    cta_strength = st.selectbox("CTA Strength", ["Weak", "Moderate", "Strong"])
    past_brand_strength = st.selectbox("Brand Strength", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("Predict Performance")


# -------------------------
# PROCESS INPUT & PREDICT
# -------------------------
if submitted:
    if not all([model, scaler, le_target]):
        st.error("❌ Model artifacts missing. Upload model files.")
        st.stop()

    # Convert categorical → numeric
    cat_map = {
        "Male": 0, "Female": 1,
        "Tier 1": 1, "Tier 2": 2, "Tier 3": 3,
        "Instagram": 1, "YouTube": 2, "Facebook": 3, "LinkedIn": 4,
        "Weak": 1, "Moderate": 2, "Strong": 3,
        "Low": 1, "Medium": 2, "High": 3
    }

    X = np.array([
        age,
        cat_map[gender],
        cat_map[city],
        budget,
        duration,
        cat_map[platform],
        creatives,
        cat_map[cta_strength],
        cat_map[past_brand_strength]
    ]).reshape(1, -1)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict class
    y_pred = model.predict(X_scaled)[0]
    label = le_target.inverse_transform([y_pred])[0]

    # Predict probabilities
    try:
        probs = model.predict_proba(X_scaled)
    except Exception:
        probs = None

    st.subheader("🎯 Prediction Result")
    st.info(f"**Predicted Performance: {label}**")

    # -------------------------
    # SAFE PROBABILITY TABLE
    # -------------------------
    if probs is not None:
        try:
            probs = np.array(probs).flatten()
            classes = list(le_target.classes_)

            if len(probs) == len(classes):
                prob_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": probs
                }).sort_values("Probability", ascending=False)

                st.subheader("📌 Prediction Probabilities")
                st.dataframe(prob_df)
            else:
                st.warning("⚠ Probability output shape mismatch, cannot display probability table.")

        except Exception as e:
            st.error(f"Failed to format probabilities: {e}")

    st.success("✅ Prediction Complete!")
