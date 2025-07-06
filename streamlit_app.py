import streamlit as st
import requests
import pandas as pd
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Hypertension Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Session State for Prediction History ---
if "history" not in st.session_state:
    st.session_state["history"] = []
if "batch_history" not in st.session_state:
    st.session_state["batch_history"] = []

st.title("🩺 Hypertension Prediction App")
st.markdown("""
This app uses a machine learning model to predict the likelihood of hypertension based on clinical features. Powered by FastAPI and Streamlit.
""")

# --- Sidebar: Model Info & Documentation ---
st.sidebar.header("Model & Feature Documentation")
if st.sidebar.button("Show Model Info"):
    try:
        resp = requests.get(f"{API_URL}/model-info")
        if resp.status_code == 200:
            info = resp.json()
            st.sidebar.success(f"Model: {info['model_type']}")
            st.sidebar.write(f"**Features ({info['num_features']}):**")
            st.sidebar.write(", ".join(info['features']))
            st.sidebar.write(f"**Target Classes:** {info['target_classes']}")
            st.sidebar.write(f"**Target Description:** {info['target_description']}")
        else:
            st.sidebar.error("Failed to fetch model info.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

with st.sidebar.expander("Feature Descriptions", expanded=False):
    st.markdown("""
    - **age**: Age of the patient (years)
    - **sex**: Sex (0 = Female, 1 = Male)
    - **cp**: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
    - **trestbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)
    - **restecg**: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (1 = Yes, 0 = No)
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **slope**: Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
    - **ca**: Number of major vessels colored by fluoroscopy (0-4)
    - **thal**: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect, 0 = unknown)
    """)

# --- Sidebar: Health Check ---
if st.sidebar.button("Check API Health"):
    try:
        resp = requests.get(f"{API_URL}/health")
        if resp.status_code == 200:
            health = resp.json()
            if health["status"] == "healthy" and health["model_loaded"]:
                st.sidebar.success("API is healthy and model is loaded.")
            else:
                st.sidebar.warning("API is running but model is not loaded.")
        else:
            st.sidebar.error("API health check failed.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# --- Main: Input Form ---
st.header("Single Patient Prediction")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0)
        sex = st.selectbox("Sex", options=[(0.0, "Female"), (1.0, "Male")], format_func=lambda x: x[1])[0]
        cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: f"{x}")
        trestbps = st.number_input("Resting BP (trestbps)", min_value=0, max_value=300, value=120)
    with col2:
        chol = st.number_input("Cholesterol (chol)", min_value=0, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=0, max_value=300, value=150)
    with col3:
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope", options=[0, 1, 2])
        ca = st.selectbox("Major Vessels (ca)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])
    submitted = st.form_submit_button("Predict Hypertension")

if submitted:
    patient = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    try:
        with st.spinner("Predicting..."):
            resp = requests.post(f"{API_URL}/predict", json=patient)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Prediction: {'Hypertension' if result['prediction'] == 1 else 'No Hypertension'}")
            st.info(f"Probability: {result['probability']:.3f}")
            st.write(f"Confidence: **{result['confidence']}**")
            # --- Visualisation: Probability Gauge ---
            st.progress(result['probability'])
            st.session_state["history"].append({"input": patient, "result": result})
        else:
            st.error(f"Prediction failed: {resp.text}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Prediction History ---
st.subheader("Prediction History (This Session)")
if st.session_state["history"]:
    for i, entry in enumerate(reversed(st.session_state["history"])):
        with st.expander(f"Prediction #{len(st.session_state['history'])-i}"):
            st.json(entry["input"])
            st.write(f"Prediction: {'Hypertension' if entry['result']['prediction'] == 1 else 'No Hypertension'}")
            st.write(f"Probability: {entry['result']['probability']:.3f}")
            st.write(f"Confidence: {entry['result']['confidence']}")
            st.progress(entry['result']['probability'])
else:
    st.info("No predictions made yet.")

# --- Batch Prediction ---
st.header("Batch Prediction (CSV Upload)")
st.markdown("Upload a CSV file with the 13 required columns for batch prediction.")
example_df = pd.DataFrame([
    {"age": 65, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 2.0, "slope": 1, "ca": 0, "thal": 3},
    {"age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2}
])
with st.expander("Show Example CSV Format"):
    st.dataframe(example_df)
    st.code(example_df.to_csv(index=False), language="csv")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        if st.button("Predict Batch"):
            if len(df) > 100:
                st.warning("Batch size too large. Maximum 100 rows allowed.")
            else:
                patients = df.to_dict(orient="records")
                try:
                    with st.spinner("Predicting batch..."):
                        resp = requests.post(f"{API_URL}/predict-batch", json=patients)
                    if resp.status_code == 200:
                        results = resp.json()
                        st.success(f"Batch prediction completed for {results['total_patients']} patients.")
                        results_df = pd.DataFrame(results["predictions"])
                        st.dataframe(results_df)
                        # --- Visualisation: Histogram of Probabilities ---
                        st.subheader("Batch Prediction Probability Distribution")
                        st.bar_chart(results_df["probability"])
                        # Save to session history
                        st.session_state["batch_history"].append({"input": df, "results": results_df})
                    else:
                        st.error(f"Batch prediction failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# --- Batch Prediction History ---
st.subheader("Batch Prediction History (This Session)")
if st.session_state["batch_history"]:
    for i, entry in enumerate(reversed(st.session_state["batch_history"])):
        with st.expander(f"Batch Prediction #{len(st.session_state['batch_history'])-i}"):
            st.write("Input Data:")
            st.dataframe(entry["input"].head())
            st.write("Results:")
            st.dataframe(entry["results"])
            st.bar_chart(entry["results"]["probability"])
else:
    st.info("No batch predictions made yet.")

st.markdown("---")
st.caption("Made with ❤️ using FastAPI, Streamlit, and scikit-learn.") 