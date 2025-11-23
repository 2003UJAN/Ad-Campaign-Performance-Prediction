# streamlit_app.py
"""
Streamlit app for Ad-Campaign Performance Prediction + Gemini insights
Place this file in the same folder as your model artifacts (app/) or run from repo root
The app will try to find artifacts in:
  1) same folder as this script
  2) ../models or ./models (convention)
"""
import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# Optional Gemini import
try:
    from google import generativeai as genai
    GEMINI_SDK = True
except Exception:
    GEMINI_SDK = False

sns.set(style="whitegrid")

# ---------- Styling ----------
def apply_theme(theme="Dark"):
    if theme == "Dark":
        css = """
        <style>
        .stApp { background-color: #0e1117; color: #E6EDF3; }
        .stSidebar { background-color: #0b0f13; color: #E6EDF3; }
        .stButton>button { background-color:#1f6feb; color:white; border-radius:6px;}
        .stMetric>div>div{color:#E6EDF3;}
        .css-1d391kg{color:#E6EDF3;} /* small label text fallback */
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

# ---------- Artifact loading ----------
@st.cache_resource
def load_artifacts():
    """
    Load model, scaler, and label encoder.
    Looks for artifacts next to this script, then in ../models and ./models
    Returns dict with keys: model, scaler, le_target, encoders (categorical LabelEncoders if available)
    """
    artifacts = {"model": None, "scaler": None, "le_target": None, "encoders": {}}

    # base_dir set to folder containing this script (works when executed by streamlit)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # candidate paths to check (script dir has priority)
    candidates = [
        base_dir,
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "..", "models"),
        os.path.join(base_dir, "..", "..", "models"),
    ]

    # file names
    model_fname = "ad_model.pkl"
    scaler_fname = "scaler.pkl"
    le_fname = "label_encoder.pkl"

    found = False
    for cand in candidates:
        model_path = os.path.join(cand, model_fname)
        scaler_path = os.path.join(cand, scaler_fname)
        le_path = os.path.join(cand, le_fname)

        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
            try:
                artifacts["model"] = joblib.load(model_path)
                artifacts["scaler"] = joblib.load(scaler_path)
                artifacts["le_target"] = joblib.load(le_path)
                found = True
                break
            except Exception as e:
                # continue searching other candidates if load fails
                st.warning(f"Found artifacts in {cand} but failed to load: {e}")
                continue

    if not found:
        # best-effort: try loading individually from script dir
        try:
            p = os.path.join(base_dir, model_fname)
            if os.path.exists(p):
                artifacts["model"] = joblib.load(p)
        except Exception as e:
            st.warning(f"Could not load ad_model.pkl from {base_dir}: {e}")
        try:
            p = os.path.join(base_dir, scaler_fname)
            if os.path.exists(p):
                artifacts["scaler"] = joblib.load(p)
        except Exception as e:
            st.warning(f"Could not load scaler.pkl from {base_dir}: {e}")
        try:
            p = os.path.join(base_dir, le_fname)
            if os.path.exists(p):
                artifacts["le_target"] = joblib.load(p)
        except Exception as e:
            st.warning(f"Could not load label_encoder.pkl from {base_dir}: {e}")

    # Try to build simple categorical encoders from a training CSV if present
    training_csv_candidates = [
        os.path.join(base_dir, "ads_synthetic.csv"),
        os.path.join(base_dir, "..", "data", "ads_synthetic.csv"),
        os.path.join(base_dir, "data", "ads_synthetic.csv"),
        os.path.join(base_dir, "..", "ads_synthetic.csv"),
    ]
    encoders = {}
    for csv_path in training_csv_candidates:
        if os.path.exists(csv_path):
            try:
                df_train = pd.read_csv(csv_path)
                for col in ["Gender", "City", "AdType"]:
                    if col in df_train.columns:
                        le = LabelEncoder()
                        le.fit(df_train[col].astype(str).values)
                        encoders[col] = le
                break
            except Exception:
                continue

    # fallback encoders
    if not encoders:
        encoders = {
            "Gender": LabelEncoder().fit(["Male", "Female", "Other"]),
            "City": LabelEncoder().fit(["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]),
            "AdType": LabelEncoder().fit(["Video", "Image", "Carousel", "Shorts"])
        }

    artifacts["encoders"] = encoders
    return artifacts

# ---------- Helpers ----------
def encode_input_df(df, encoders):
    """Encode expected categorical columns using provided LabelEncoders.
       Unknown categories mapped to nearest (0) fallback."""
    df2 = df.copy()
    for c in ["Gender", "City", "AdType"]:
        if c in df2.columns:
            try:
                df2[c] = encoders[c].transform(df2[c].astype(str).values)
            except Exception:
                # map unseen values to most common class (index 0)
                df2[c] = 0
    return df2

def plot_feature_importance(importances, feature_names):
    idx = np.argsort(importances)[::-1]
    sorted_feats = [feature_names[i] for i in idx]
    sorted_imps = importances[idx]
    fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names)*0.35)))
    ax.barh(sorted_feats[::-1], sorted_imps[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

def plot_radar(values, labels, title="Feature Radar"):
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    vals = list(values) + [values[0]]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels)
    ax.plot(angles, vals, linewidth=2, linestyle='solid')
    ax.fill(angles, vals, alpha=0.25)
    ax.set_title(title)
    st.pyplot(fig)

def call_gemini_insights(features: dict, prediction_label: str):
    if not GEMINI_SDK:
        return "Gemini SDK not installed; install google-generativeai to enable generative insights."
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY not set; set environment variable to enable generative insights."
    try:
        genai.configure(api_key=api_key)
        prompt = f"""
You are an advertising insights analyst.

Input features: {features}
Model prediction: {prediction_label}

Task:
1) Explain in 3 concise bullets why the model predicted {prediction_label}.
2) Provide 3 actionable suggestions (<=12 words each) to improve the ad.
3) Suggest 2 short alternative headlines (<=8 words each).
Keep tone business-friendly.
"""
        resp = genai.generate_text(model="gemini-2.0-flash-lite", prompt=prompt, max_output_tokens=300, temperature=0.2)
        return resp.text
    except Exception as e:
        return f"Gemini call failed: {e}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AdPerf — Predictor", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 AdPerf — Ad Performance Predictor + Insights")

# Sidebar
with st.sidebar:
    st.header("Controls")
    theme = st.radio("Theme", ["Dark", "Light"], index=0)
    apply_theme(theme)
    st.markdown("---")
    st.write("Model artifact detection (auto)")
    st.write("Place `ad_model.pkl`, `scaler.pkl`, `label_encoder.pkl` next to this app or in a `models/` folder.")
    st.markdown("---")
    st.write("Optional: enable Gemini generative insights")
    st.write(f"Gemini SDK installed: {GEMINI_SDK}")
    st.write("Set GEMINI_API_KEY as an env var to enable.")

# Load artifacts
art = load_artifacts()
model = art.get("model")
scaler = art.get("scaler")
le_target = art.get("le_target")
encoders = art.get("encoders", {})

# Show artifact load status
col_status_1, col_status_2, col_status_3 = st.columns(3)
with col_status_1:
    if model is None:
        st.error("Model: ❌ not loaded")
    else:
        st.success("Model: ✔ loaded")
with col_status_2:
    if scaler is None:
        st.error("Scaler: ❌ not loaded")
    else:
        st.success("Scaler: ✔ loaded")
with col_status_3:
    if le_target is None:
        st.error("LabelEncoder: ❌ not loaded")
    else:
        st.success("LabelEncoder: ✔ loaded")

st.markdown("---")

# Main layout: left input / right results
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input data")
    st.info("Upload a CSV matching training columns or enter a single row manually.")
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    df_input = None
    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
            st.success(f"Loaded {uploaded.name} — {df_input.shape[0]} rows")
            st.dataframe(df_input.head())
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            df_input = None

    st.markdown("Or enter a single ad row:")
    with st.form("manual_form"):
        Age = st.number_input("Age", min_value=15, max_value=100, value=30)
        Gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        City = st.selectbox("City", options=["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"])
        AdType = st.selectbox("Ad Type", options=["Video", "Image", "Carousel", "Shorts"])
        Budget = st.number_input("Budget (INR)", min_value=0, max_value=5_000_000, value=50000, step=1000)
        DurationSec = st.number_input("Duration (sec)", min_value=1, max_value=600, value=30)
        Impressions = st.number_input("Impressions", min_value=0, max_value=50_000_000, value=200000, step=1000)
        CTR = st.number_input("CTR (%)", min_value=0.0, max_value=100.0, value=1.5, format="%.3f")
        Clicks = st.number_input("Clicks", min_value=0, max_value=10_000_000, value=int(Impressions*(CTR/100)))
        ConversionRate = st.number_input("Conversion Rate (%)", min_value=0.0, max_value=100.0, value=0.5, format="%.3f")
        Conversions = st.number_input("Conversions", min_value=0, max_value=1_000_000, value=int(Clicks*(ConversionRate/100)))
        submit = st.form_submit_button("Use manual row")
        if submit:
            df_input = pd.DataFrame([{
                "Age": Age, "Gender": Gender, "City": City, "AdType": AdType,
                "Budget": Budget, "DurationSec": DurationSec, "Impressions": Impressions,
                "CTR": CTR, "Clicks": Clicks, "ConversionRate": ConversionRate, "Conversions": Conversions
            }])
            st.success("Manual row prepared for prediction")
            st.dataframe(df_input)

with col2:
    st.subheader("Prediction")
    if model is None or scaler is None or le_target is None:
        st.warning("Model artifacts missing — predictions will not run until artifacts are loaded.")
    run_pred = st.button("Run Prediction")

# Prediction execution
if run_pred:
    if df_input is None:
        st.error("No input available. Upload CSV or fill manual row.")
    else:
        # verify expected columns (best-effort)
        expected_cols = ["Age","Gender","City","AdType","Budget","DurationSec","Impressions","CTR","Clicks","ConversionRate","Conversions"]
        missing = [c for c in expected_cols if c not in df_input.columns]
        if missing:
            st.warning(f"Input is missing expected columns: {missing}. Attempting to continue with available columns.")
        # encode categorical
        df_proc = df_input.copy()
        df_proc = df_proc.reindex(columns=expected_cols, fill_value=0)

        df_proc = encode_input_df(df_proc, encoders)

        # prepare numeric matrix
        try:
            X = df_proc[expected_cols].astype(float).values
        except Exception as e:
            st.error(f"Failed to prepare numeric input: {e}")
            X = None

        if X is not None:
            try:
                Xs = scaler.transform(X)
            except NotFittedError:
                st.warning("Scaler not fitted. Attempting to fit scaler on input (not recommended).")
                try:
                    scaler.fit(X)
                    Xs = scaler.transform(X)
                except Exception as e:
                    st.error(f"Scaler fit/transform failed: {e}")
                    Xs = None
            except Exception as e:
                st.error(f"Scaler.transform failed: {e}")
                Xs = None

            if Xs is not None:
                try:
                    preds = model.predict(Xs)
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    preds = None

                probs = None
                try:
                    probs = model.predict_proba(Xs)
                except Exception:
                    # some models may not have predict_proba
                    probs = None

                if preds is not None:
                    pred_labels = le_target.inverse_transform(preds.astype(int))
                    st.metric("Predicted Performance", str(pred_labels[0]))

                    if probs is not None:
                        classes = le_target.classes_
                        prob_df = pd.DataFrame([probs[0]], columns=classes).T.rename(columns={0:"probability"})
                        st.subheader("Prediction Probabilities")
                        st.write(prob_df)

                    # show feature importance if available
                    st.subheader("Feature Importance")
                    if hasattr(model, "feature_importances_"):
                        feature_names = expected_cols
                        imps = model.feature_importances_
                        plot_feature_importance(imps, feature_names)
                    else:
                        st.info("Feature importance not available for this model type.")

                    # radar visualization for numeric subset
                    st.subheader("Normalized Feature Radar")
                    radar_feats = ["Budget","Impressions","CTR","Clicks","ConversionRate","Conversions"]
                    radar_vals = []
                    row0 = df_input.iloc[0]
                    for f in radar_feats:
                        v = float(row0[f]) if f in df_input.columns else 0.0
                        # normalized heuristics
                        if f == "Budget":
                            radar_vals.append(min(v/200000.0, 1.0))
                        elif f == "Impressions":
                            radar_vals.append(min(v/1_000_000.0, 1.0))
                        elif f == "CTR":
                            radar_vals.append(min(v/10.0, 1.0))
                        elif f == "Clicks":
                            radar_vals.append(min(v/50000.0, 1.0))
                        elif f == "ConversionRate":
                            radar_vals.append(min(v/5.0, 1.0))
                        elif f == "Conversions":
                            radar_vals.append(min(v/5000.0, 1.0))
                        else:
                            radar_vals.append(0.0)
                    plot_radar(radar_vals, radar_feats, title="Normalized Numeric Radar")

                    # Gemini optional insights
                    st.subheader("Generative Insights (Gemini)")
                    if GEMINI_SDK and os.getenv("GEMINI_API_KEY"):
                        with st.spinner("Generating insights..."):
                            insight = call_gemini_insights(df_input.iloc[0].to_dict(), str(pred_labels[0]))
                            st.markdown(insight.replace("\n", "  \n"))
                    else:
                        if not GEMINI_SDK:
                            st.info("Gemini SDK not installed. Skip generative insights.")
                        else:
                            st.info("GEMINI_API_KEY not set. Add env var to enable generative insights.")

                # If batch CSV input and predictions available, offer download
                if df_input.shape[0] > 1 and preds is not None:
                    try:
                        # decode all preds to labels
                        all_labels = le_target.inverse_transform(preds.astype(int))
                        out_df = df_input.copy()
                        out_df["Predicted_Performance"] = all_labels
                        if probs is not None and probs.shape[1] == len(le_target.classes_):
                            for i, cls in enumerate(le_target.classes_):
                                out_df[f"prob_{cls}"] = probs[:, i] if probs.ndim>1 else probs[0][i]
                        # prepare csv download
                        csv_buf = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Predictions (CSV)", csv_buf, file_name="predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.warning(f"Could not prepare batch download: {e}")

# Footer
st.markdown("---")
st.caption("Built with ♥ for ad campaign prototyping. Put model artifacts in the same folder as this script or in models/ directory.")
