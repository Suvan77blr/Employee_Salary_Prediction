import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Expected input columns
model_columns = [
    'age', 'workclass', 'educational-num', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'hours-per-week', 'native-country',
    'capital-gain-log', 'has-capital-gain', 'has-capital-loss',
    'capital-loss-log', 'hours-per-week-scaled'
]

# Page config
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #4A7EBB;'>Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Will your income be >50K? Let's find out!</h4>", unsafe_allow_html=True)
st.markdown("---")

# Name input
name = st.text_input("Enter your name:", value="User")

# Grouped layout
with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 30)
        education_num = st.selectbox("Education Level (Num)", list(range(1, 17)))
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

    with col2:
        workclass = st.selectbox("Workclass", [
            "Private", "Local-gov", "Others", "Self-emp-not-inc",
            "Federal-gov", "State-gov", "Self-emp-inc"
        ])

        marital_status = st.selectbox("Marital Status", [
            "Single", "Married", "Previously Married"
        ])

        occupation = st.selectbox("Occupation", [
            "Blue-Collar", "Agriculture", "Service", "Unknown", "White-Collar", "Other"
        ])

        relationship = st.selectbox("Relationship", [
            "Own-child", "Spouse", "Not-in-family", "Unmarried", "Other-relative"
        ])

        race = st.selectbox("Race", [
            "Black", "White", "Asian-Pac-Islander", "Other", "Amer-Indian-Eskimo"
        ])

        gender = st.radio("Gender", ["Male", "Female"])

        native_country = st.selectbox("Native Country", [
            "North America", "Unknown", "South America", "Central America", "Caribbean",
            "Europe", "Asia (East)", "Other", "Asia (South)", "Asia (West)"
        ])

    # Submit button
    submitted = st.form_submit_button("Predict Income")

if submitted:
    # Build raw input DataFrame
    raw_input = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "educational-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }])

    # Label encoding
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        raw_input[col] = encoders[col].transform(raw_input[col])

    # Derived features
    raw_input["capital-gain-log"] = np.log1p(raw_input["capital-gain"])
    raw_input["has-capital-gain"] = raw_input["capital-gain"].apply(lambda x: 1 if x > 0 else 0)
    raw_input["capital-loss-log"] = np.log1p(raw_input["capital-loss"])
    raw_input["has-capital-loss"] = raw_input["capital-loss"].apply(lambda x: 1 if x > 0 else 0)
    raw_input["hours-per-week-scaled"] = (raw_input["hours-per-week"] - 40) / 40

    # Drop raw capital features
    raw_input.drop(columns=["capital-gain", "capital-loss"], inplace=True)

    # Reorder columns
    input_data = raw_input[model_columns]

    # Predict
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Result section
    st.markdown("---")
    st.markdown(f"### Hai there, **{name}**!")
    st.markdown(f"### Based on your input:")

    if pred == ">50K":
        st.success("**Predicted Income: >50K**")
        st.markdown(f"Confidence: **{prob:.2%}**")
        st.progress(prob)
    else:
        st.info("**Predicted Income: <=50K**")
        st.markdown(f"Confidence: **{(1 - prob):.2%}**")
        st.progress(1 - prob)
