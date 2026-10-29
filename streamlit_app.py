
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ============================================
# LOAD TRAINED MODEL AND SCALER
# ============================================

@st.cache_resource
def load_model_files():
    model_path = Path(__file__).parent / "heart_disease_model.pkl"
    scaler_path = Path(__file__).parent / "heart_disease_scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler

try:
    model, scaler = load_model_files()
    st.success("Prediction model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# Features must match the training order exactly
FEATURES = [
    "Age",
    "Gender",
    "Smoking",
    "BMI",
    "Heart rate",
    "blood Oxygen",
    "body temperature",
    "Systolic_BP",
    "Diastolic_BP",
    "ECG_Raw_ADC_Demo"
]

if len(FEATURES) != model.n_features_in_:
    st.error(
        f"Feature mismatch: model expects "
        f"{model.n_features_in_} features, "
        f"but the app provides {len(FEATURES)}."
    )
    st.stop()

if hasattr(scaler, "n_features_in_"):
    if scaler.n_features_in_ != len(FEATURES):
        st.error("The scaler does not match the 10 input features.")
        st.stop()

# ============================================
# PAGE HEADING
# ============================================

st.title("❤️ Heart Disease Prediction System")
st.write("Machine Learning-Based Heart Disease Prediction System")

st.sidebar.title("About the Project")
st.sidebar.write(
    "An academic project using Random Forest "
    "and patient health parameters."
)
st.sidebar.caption(
    "Experimental prototype — not a medical diagnosis."
)

st.divider()
st.subheader("Enter Health Parameters")

# ============================================
# INPUT FORM
# ============================================

with st.form("health_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)", 1, 120, 25
        )

        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        smoking = st.selectbox(
            "Smoking Status", ["No", "Yes"]
        )

        bmi = st.number_input(
            "BMI", 1.0, 100.0, 22.0
        )

        heart_rate = st.number_input(
            "Heart Rate (BPM)", 1, 300, 75
        )

    with col2:
        blood_oxygen = st.number_input(
            "Blood Oxygen (%)", 0.0, 100.0, 98.0
        )

        body_temperature = st.number_input(
            "Body Temperature (°C)", 25.0, 45.0, 36.8
        )

        systolic_bp = st.number_input(
            "Systolic Blood Pressure", 0, 300, 120
        )

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure", 0, 200, 80
        )

        ecg_reading = st.number_input(
            "ECG Raw ADC Reading", 0, value=500
        )

    submitted = st.form_submit_button(
        "Predict Heart Disease",
        type="primary",
        use_container_width=True
    )

# ============================================
# PREDICTION
# ============================================

if submitted:
    # Match the encoding used during training:
    # Male = 0, Female = 1
    # No smoking = 0, Yes smoking = 1

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": 1 if gender == "Female" else 0,
        "Smoking": 1 if smoking == "Yes" else 0,
        "BMI": bmi,
        "Heart rate": heart_rate,
        "blood Oxygen": blood_oxygen,
        "body temperature": body_temperature,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "ECG_Raw_ADC_Demo": ecg_reading
    }], columns=FEATURES)

    try:
        # Apply the SAME fitted scaler used during training
        scaled_input = scaler.transform(input_data)

        # Predict using the trained model
        prediction = int(model.predict(scaled_input)[0])

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "Model prediction: HeartProblem = 1 "
                "(positive class)"
            )
        else:
            st.success(
                "Model prediction: HeartProblem = 0 "
                "(negative class)"
            )

        st.caption(
            "This is a machine-learning classification result, "
            "not a confirmed diagnosis. Seek professional medical "
            "assessment for health concerns."
        )

        with st.expander("View input values used for prediction"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
