
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================
# PAGE CONFIGURATION
# ============================================

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
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# ============================================
# FEATURES
# Must match the training order exactly
# ============================================

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
# DASHBOARD HEADING
# ============================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    "### Machine Learning-Based Heart Disease Prediction System"
)

st.markdown(
    """
    Enter the patient's health parameters below to generate
    a prediction using the trained Random Forest model.
    """
)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📌 About the Project")

st.sidebar.write(
    """
    This academic project uses a trained Random Forest
    machine learning model to classify input parameters
    into the model's two target classes.
    """
)

st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Classes:** 0 and 1")
st.sidebar.markdown("**Inputs:** 10 health parameters")

st.sidebar.info(
    "Educational prototype only. This application is not "
    "a medical diagnostic tool."
)

st.divider()

# ============================================
# INPUT FORM
# ============================================

st.subheader("🩺 Enter Health Parameters")

with st.form("health_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Personal Information")

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=25
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        smoking = st.selectbox(
            "Smoking Status",
            ["No", "Yes"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=1.0,
            max_value=100.0,
            value=22.0,
            step=0.1
        )

        heart_rate = st.number_input(
            "Heart Rate (BPM)",
            min_value=1,
            max_value=300,
            value=75
        )

    with col2:
        st.markdown("#### 💓 Vital Signs and ECG")

        blood_oxygen = st.number_input(
            "Blood Oxygen (%)",
            min_value=0.0,
            max_value=100.0,
            value=98.0,
            step=0.1
        )

        body_temperature = st.number_input(
            "Body Temperature (°C)",
            min_value=25.0,
            max_value=45.0,
            value=36.8,
            step=0.1
        )

        systolic_bp = st.number_input(
            "Systolic Blood Pressure",
            min_value=0,
            max_value=300,
            value=120
        )

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure",
            min_value=0,
            max_value=200,
            value=80
        )

        ecg_reading = st.number_input(
            "ECG Raw ADC Reading",
            min_value=0,
            value=500
        )

    submitted = st.form_submit_button(
        "🔍 Predict Heart Disease",
        type="primary",
        use_container_width=True
    )

# ============================================
# PREDICTION RESULT DASHBOARD
# ============================================

if submitted:

    # Use the same encoding used during training:
    # Male = 0, Female = 1
    # No smoking = 0, Yes smoking = 1

    input_data = pd.DataFrame(
        [{
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
        }],
        columns=FEATURES
    )

    try:
        # Apply the fitted scaler used during training
        scaled_input = scaler.transform(input_data)

        # Generate prediction
        prediction = int(model.predict(scaled_input)[0])

        st.divider()
        st.subheader("📊 Prediction Results")

        # ------------------------------------
        # NEGATIVE PREDICTION
        # ------------------------------------

        if prediction == 0:

            st.success("💚 No Heart Disease Detected")

            st.markdown(
                """
                <div style="
                    padding: 22px;
                    border-radius: 12px;
                    background-color: #e8f5e9;
                    border-left: 6px solid #2e7d32;
                    color: #1b5e20;
                    margin: 12px 0;
                ">
                    <h2 style="margin: 0;">
                        Negative Prediction
                    </h2>
                    <p style="font-size: 17px;">
                        The model did not predict heart disease
                        for the entered parameters.
                    </p>
                    <b>Model Output: 0</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ------------------------------------
        # POSITIVE PREDICTION
        # ------------------------------------

        else:

            st.warning("⚠️ Heart Disease Indicated")

            st.markdown(
                """
                <div style="
                    padding: 22px;
                    border-radius: 12px;
                    background-color: #ffebee;
                    border-left: 6px solid #c62828;
                    color: #b71c1c;
                    margin: 12px 0;
                ">
                    <h2 style="margin: 0;">
                        Positive Prediction
                    </h2>
                    <p style="font-size: 17px;">
                        The model flagged a possible indication
                        of heart disease in the entered parameters.
                        Professional medical evaluation is advised.
                    </p>
                    <b>Model Output: 1</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        
        # ------------------------------------
        # MODEL CLASS PROBABILITY
        # ------------------------------------

       # if hasattr(model, "predict_proba"):

        #    probabilities = model.predict_proba(scaled_input)[0]
         #   classes = list(model.classes_)

          #  if 1 in classes:

           #     positive_probability = float(
            #        probabilities[classes.index(1)]
             #   )

              #  st.subheader("📊 Model Probability")

               # st.progress(positive_probability)

                #st.markdown(
                 #   f"**Estimated probability for class 1:** "
                  #  f"{positive_probability:.1%}"
               # )

                #st.caption(
                 #   "This value represents the model's estimated "
                  #  "probability for class 1. It is not a clinically "
                 #   "validated estimate of an individual's heart "
                 #   "disease risk."
             #   )

        # ------------------------------------
        # INPUT SUMMARY
        # ------------------------------------

        with st.expander(
            "📋 View Input Parameters Used for Prediction"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )

        # ------------------------------------
        # MEDICAL DISCLAIMER
        # ------------------------------------

        st.divider()

        st.caption(
            "⚕️ Disclaimer: This is an educational machine learning "
            "prototype, not a medical diagnosis. A negative result "
            "does not rule out heart disease, and a positive result "
            "does not confirm it. Consult a qualified healthcare "
            "professional for medical concerns."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "Heart Disease Prediction System | "
    "Random Forest Machine Learning | Academic Project"
)
