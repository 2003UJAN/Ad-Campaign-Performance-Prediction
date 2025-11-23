import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Optional Gemini
try:
    from google import generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Ad Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# DARK / LIGHT THEME
# -------------------------
theme = st.sidebar.radio("🌗 Theme Mode", ["Dark", "Light"], index=0)

if theme == "Dark":
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #E6EDF3; }
        .stButton>button { background-color:#1f6feb; color:white; border-radius:6px; }
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div {
            background-color:#161b22;
            color:#E6EDF3;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; color: #111827; }
        .stButton>button { background-color:#2563EB; color:white; border-radius:6px; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# LOAD ARTIFACTS FROM /app ONLY
# -------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "ad_model.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    label_path = os.path.join(base_dir, "label_encoder.pkl")

    artifacts = {
        "model": joblib.load(model_path) if os.path.exists(model_path) else None,
        "scaler": joblib.load(scaler_path) if os.path.exists(scaler_path) else None,
        "encoder": joblib.load(label_path) if os.path.exists(label_path) else None
    }
    return artifacts

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
encoder = artifacts["encoder"]

# Status
if not all([model, scaler, encoder]):
    st.error("❌ Model files not found in /app. Ensure `.pkl` files are next to app.py.")
else:
    st.success("✅ Model Loaded Successfully!")

st.title("📊 Ad Campaign Performance Predictor")
st.write("Predict **High / Medium / Low** performance using campaign features.")

# Expected features
FEATURES = [
    "Age","Gender","City","AdType",
    "Budget","DurationSec","Impressions","CTR",
    "Clicks","ConversionRate","Conversions"
]

# Encoding maps (same as training)
GENDERS = ["Female","Male"]
CITIES = ["Bangalore","Chennai","Delhi","Hyderabad","Kolkata","Mumbai"]
ADTYPES = ["Carousel","Image","Shorts","Video"]

def encode(df):
    df = df.copy()
    df["Gender"] = df["Gender"].map({v:i for i,v in enumerate(GENDERS)})
    df["City"] = df["City"].map({v:i for i,v in enumerate(CITIES)})
    df["AdType"] = df["AdType"].map({v:i for i,v in enumerate(ADTYPES)})
    return df

# -------------------------
# INPUT SECTION
# -------------------------
left, right = st.columns([1.3,1])

with left:
    st.subheader("📂 Upload CSV File (Batch Prediction)")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    df_input = None

    if csv_file:
        try:
            df_input = pd.read_csv(csv_file)
            st.success(f"Loaded {csv_file.name} — {df_input.shape[0]} rows")
            st.dataframe(df_input.head())
        except:
            st.error("Invalid CSV format.")

    st.markdown("---")
    st.subheader("✏ Manual Single Input")

    with st.form("manual"):
        Age = st.number_input("Age", 18, 65, 30)
        Gender = st.selectbox("Gender", GENDERS)
        City = st.selectbox("City", CITIES)
        AdType = st.selectbox("Ad Type", ADTYPES)
        Budget = st.number_input("Budget (₹)", 1000, 2000000, 50000)
        DurationSec = st.number_input("Duration (sec)", 5, 180, 30)
        Impressions = st.number_input("Impressions", 0, 10000000, 200000)
        CTR = st.number_input("CTR (%)", 0.0, 100.0, 1.5)
        Clicks = st.number_input("Clicks", 0, 1000000, int(Impressions*CTR/100))
        ConversionRate = st.number_input("Conversion Rate (%)", 0.0, 100.0, 0.8)
        Conversions = st.number_input("Conversions", 0, 1000000, int(Clicks*ConversionRate/100))
        submit = st.form_submit_button("Use this input")

        if submit:
            df_input = pd.DataFrame([{
                "Age":Age,"Gender":Gender,"City":City,"AdType":AdType,
                "Budget":Budget,"DurationSec":DurationSec,"Impressions":Impressions,
                "CTR":CTR,"Clicks":Clicks,"ConversionRate":ConversionRate,"Conversions":Conversions
            }])
            st.success("✅ Manual Record Loaded")
            st.dataframe(df_input)

# -------------------------
# PREDICTION
# -------------------------
with right:
    st.subheader("🎯 Prediction Output")

    if df_input is not None and all([model, scaler, encoder]):
        if st.button("Run Prediction"):
            # Ensure correct columns
            df_proc = df_input.copy()
            df_proc = encode(df_proc)
            df_proc = df_proc[FEATURES]

            X = df_proc.astype(float).values
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled)
            labels = encoder.inverse_transform(preds)

            st.success(f"📌 Predicted Performance: **{labels[0]}**")

            # Probabilities
            try:
                probs = np.array(model.predict_proba(X_scaled)[0]).flatten()
                classes = list(encoder.classes_)
                prob_df = pd.DataFrame({"Class":classes,"Probability":probs}).sort_values("Probability",ascending=False)
                st.subheader("📊 Prediction Probabilities")
                st.dataframe(prob_df)
            except:
                st.warning("Model does not support predict_proba()")

            # Batch download
            if len(df_input) > 1:
                df_out = df_input.copy()
                df_out["Predicted_Performance"] = labels
                st.download_button(
                    "Download Predictions CSV",
                    df_out.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )

            # Gemini Insights
            st.subheader("🤖 Gemini Insights")
            if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                prompt = f"""
                Ad Performance Prediction: {labels[0]}
                Features: {df_input.iloc[0].to_dict()}

                Provide:
                1. Three reasons for this prediction
                2. Three improvement suggestions
                3. One catchy ad-line idea
                Keep output concise.
                """
                try:
                    response = genai.generate_text(
                        model="gemini-2.0-flash-lite",
                        prompt=prompt
                    )
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Gemini Error: {e}")
            else:
                st.info("Set GEMINI_API_KEY to enable insights.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ | Ad Campaign Performance Predictor")
