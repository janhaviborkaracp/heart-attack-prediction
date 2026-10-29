
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Heart Health Dashboard",
    page_icon="❤️",
    layout="wide"
)

# Page heading
st.title("❤️ Heart Attack Prediction System")
st.write(
    "Machine Learning-Based Heart Health Monitoring Dashboard"
)

st.info(
    "Enter the health parameters below. "
    "The trained prediction model will be connected separately."
)

st.divider()

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.write(
    "An academic project exploring machine learning "
    "and health measurements."
)

st.sidebar.caption(
    "Experimental prototype — not a medical diagnosis."
)

# Input form
st.subheader("Enter Health Parameters")

with st.form("health_form"):

    col1, col2 = st.columns(2)

    with col1:
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
            value=22.0
        )

        heart_rate = st.number_input(
            "Heart Rate (BPM)",
            min_value=1,
            max_value=300,
            value=75
        )

    with col2:
        blood_oxygen = st.number_input(
            "Blood Oxygen (%)",
            min_value=0.0,
            max_value=100.0,
            value=98.0
        )

        body_temperature = st.number_input(
            "Body Temperature (°C)",
            min_value=25.0,
            max_value=45.0,
            value=36.8
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
        "Submit Health Parameters",
        type="primary",
        use_container_width=True
    )

if submitted:
    st.subheader("Entered Health Parameters")

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Smoking": smoking,
        "BMI": bmi,
        "Heart rate": heart_rate,
        "blood Oxygen": blood_oxygen,
        "body temperature": body_temperature,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        "ECG_Raw_ADC_Demo": ecg_reading
    }])

    st.dataframe(input_data, use_container_width=True)

    st.warning(
        "Your inputs have been recorded in this session. "
        "A genuine prediction cannot be displayed until "
        "the trained model is connected."
    )

st.divider()

st.caption(
    "Academic prototype only. This application does not "
    "provide a validated heart attack diagnosis."
)
