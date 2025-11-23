import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Ad Campaign Performance Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# THEME TOGGLE (LIGHT / DARK)
# ============================================================
mode = st.sidebar.radio("🌗 Theme", ["Dark", "Light"], index=0)

if mode == "Dark":
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #E6EDF3; }
        .stSidebar { background-color: #0b0f13 !important; }
        .stButton>button {
            background-color:#1f6feb;
            color:white;
            border-radius:6px;
        }
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
        .stApp { background-color: #f8f9fa; color: #111827; }
        .stSidebar { background-color: #f3f4f6 !important; }
        .stButton>button {
            background-color:#2563EB;
            color:white;
            border-radius:6px;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# MODEL LOADER – SEARCH IN app/, ../models, ./models, root
# ============================================================
@st.cache_resource
def load_artifacts():
    """
    Load ad_model.pkl, scaler.pkl, label_encoder.pkl
    from common locations:
      - same folder as this script (app/)
      - ../models
      - ./models
      - repo root
    """
    artifacts = {"model": None, "scaler": None, "le_target": None}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        base_dir,                                # app/
        os.path.join(base_dir, "models"),        # app/models (if exists)
        os.path.join(base_dir, "..", "models"),  # ../models
        os.path.join(base_dir, ".."),            # repo root
    ]

    def _load_file(fname):
        for p in search_paths:
            path = os.path.join(p, fname)
            if os.path.exists(path):
                return joblib.load(path)
        return None

    artifacts["model"] = _load_file("ad_model.pkl")
    artifacts["scaler"] = _load_file("scaler.pkl")
    artifacts["le_target"] = _load_file("label_encoder.pkl")

    return artifacts


artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
le_target = artifacts["le_target"]

# Status panel
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.write("🧠 Model:", "✅ Loaded" if model is not None else "❌ Missing")
with col_b:
    st.write("📏 Scaler:", "✅ Loaded" if scaler is not None else "❌ Missing")
with col_c:
    st.write("🏷 Target Encoder:", "✅ Loaded" if le_target is not None else "❌ Missing")

if not all([model, scaler, le_target]):
    st.warning("⚠ Model artifacts not fully loaded. Make sure `ad_model.pkl`, `scaler.pkl`, `label_encoder.pkl` are in `app/` or `models/`.")
else:
    st.success("✅ All artifacts loaded successfully!")

st.markdown("---")

# ============================================================
# EXPECTED FEATURES (MATCH TRAINING NOTEBOOK 03)
# ads_synthetic.csv columns:
# Age, Gender, City, AdType, Budget, DurationSec,
# Impressions, CTR, Clicks, ConversionRate, Conversions, Performance
# ============================================================
FEATURE_COLS = [
    "Age", "Gender", "City", "AdType",
    "Budget", "DurationSec", "Impressions", "CTR",
    "Clicks", "ConversionRate", "Conversions"
]

# Rebuild the categorical encoding in the SAME way as training:
# LabelEncoder sorts classes alphabetically, so we replicate that.

GENDER_CLASSES = ["Female", "Male"]  # alphabetical
CITY_CLASSES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
ADTYPE_CLASSES = ["Carousel", "Image", "Shorts", "Video"]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Gender, City, AdType to integers consistent with training."""
    df_enc = df.copy()

    gender_map = {cls: idx for idx, cls in enumerate(GENDER_CLASSES)}
    city_map = {cls: idx for idx, cls in enumerate(CITY_CLASSES)}
    adtype_map = {cls: idx for idx, cls in enumerate(ADTYPE_CLASSES)}

    # Gender
    if "Gender" in df_enc.columns:
        df_enc["Gender"] = df_enc["Gender"].map(
            lambda x: gender_map.get(str(x), 0)
        )

    # City
    if "City" in df_enc.columns:
        df_enc["City"] = df_enc["City"].map(
            lambda x: city_map.get(str(x), 0)
        )

    # AdType
    if "AdType" in df_enc.columns:
        df_enc["AdType"] = df_enc["AdType"].map(
            lambda x: adtype_map.get(str(x), 0)
        )

    return df_enc


# ============================================================
# PAGE HEADER
# ============================================================
st.title("📊 Ad Campaign Performance Prediction Dashboard")
st.write(
    "Predict whether an ad campaign will perform **High / Medium / Low** "
    "based on synthetic survey and engagement features."
)

# ============================================================
# LAYOUT: LEFT (INPUT) / RIGHT (OUTPUT)
# ============================================================
left, right = st.columns([1.4, 1.1])

# ============================================================
# LEFT: FILE UPLOAD + MANUAL INPUT
# ============================================================
with left:
    st.subheader("1️⃣ Input Data")

    st.markdown("**Option A: Upload CSV** (with columns matching training data)")
    uploaded_file = st.file_uploader("Upload ads_synthetic-like CSV", type=["csv"])

    df_input = None

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` ({df_input.shape[0]} rows).")
            st.dataframe(df_input.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df_input = None

    st.markdown("---")
    st.markdown("**Option B: Enter a single ad record manually**")

    with st.form("manual_form"):
        Age = st.number_input("Age", min_value=18, max_value=65, value=30)
        Gender = st.selectbox("Gender", ["Female", "Male"])
        City = st.selectbox("City", CITY_CLASSES)
        AdType = st.selectbox("Ad Type", ADTYPE_CLASSES)
        Budget = st.number_input("Budget (₹)", min_value=1000, max_value=2000000, value=50000, step=1000)
        DurationSec = st.number_input("Ad Duration (sec)", min_value=5, max_value=180, value=30)
        Impressions = st.number_input("Impressions", min_value=0, max_value=10000000, value=200000, step=1000)
        CTR = st.number_input("CTR (%)", min_value=0.0, max_value=100.0, value=1.5, format="%.2f")
        Clicks = st.number_input("Clicks", min_value=0, max_value=1000000, value=int(Impressions * CTR / 100))
        ConversionRate = st.number_input("Conversion Rate (%)", min_value=0.0, max_value=100.0, value=0.8, format="%.2f")
        Conversions = st.number_input("Conversions", min_value=0, max_value=1000000, value=int(Clicks * ConversionRate / 100))

        use_manual = st.form_submit_button("Use this manual input")

        if use_manual:
            df_input = pd.DataFrame([{
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
            }])
            st.success("Manual record prepared.")
            st.dataframe(df_input)

# ============================================================
# RIGHT: PREDICTION OUTPUT
# ============================================================
with right:
    st.subheader("2️⃣ Prediction")

    if df_input is None:
        st.info("Upload a CSV or fill the manual form to see predictions.")
    elif not all([model, scaler, le_target]):
        st.error("❌ Model artifacts missing. Cannot run predictions.")
    else:
        if st.button("Run Prediction"):
            # Ensure df_input has all required columns
            missing_cols = [c for c in FEATURE_COLS if c not in df_input.columns]
            if missing_cols:
                st.error(f"Input missing columns: {missing_cols}")
            else:
                # Reorder and encode categoricals
                df_proc = df_input[FEATURE_COLS].copy()
                df_proc = encode_categoricals(df_proc)

                # Convert to numeric and scale
                try:
                    X = df_proc.astype(float).values
                    X_scaled = scaler.transform(X)
                except Exception as e:
                    st.error(f"Error scaling input: {e}")
                    X_scaled = None

                if X_scaled is not None:
                    # Predict classes
                    try:
                        preds = model.predict(X_scaled)
                    except Exception as e:
                        st.error(f"Error during model.predict: {e}")
                        preds = None

                    # Predict probabilities
                    probs = None
                    if preds is not None:
                        try:
                            probs = model.predict_proba(X_scaled)
                        except Exception:
                            probs = None

                    if preds is not None:
                        # Decode labels
                        labels = le_target.inverse_transform(preds.astype(int))
                        st.success(f"🎯 Predicted Performance (first row): **{labels[0]}**")

                        # Show probabilities for first row
                        if probs is not None:
                            try:
                                prob_vec = np.array(probs[0]).flatten()
                                classes = list(le_target.classes_)

                                if len(prob_vec) == len(classes):
                                    prob_df = pd.DataFrame({
                                        "Class": classes,
                                        "Probability": prob_vec
                                    }).sort_values("Probability", ascending=False)
                                    st.markdown("**Class Probabilities (first row):**")
                                    st.dataframe(prob_df)
                                else:
                                    st.warning("Probability dimension mismatch; cannot display probabilities.")
                            except Exception as e:
                                st.error(f"Error formatting probabilities: {e}")

                        # If batch input, show and enable download
                        if df_input.shape[0] > 1:
                            try:
                                all_labels = le_target.inverse_transform(preds.astype(int))
                                out_df = df_input.copy()
                                out_df["Predicted_Performance"] = all_labels

                                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                                st.markdown("**Batch predictions ready.**")
                                st.download_button(
                                    "Download predictions as CSV",
                                    data=csv_bytes,
                                    file_name="ad_performance_predictions.csv",
                                    mime="text/csv"
                                )
                                st.dataframe(out_df.head())
                            except Exception as e:
                                st.warning(f"Could not prepare batch results for download: {e}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Ad Campaign Performance Predictor • Uses pre-trained model and scaler from `app/` or `models/`.")
