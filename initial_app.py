import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained pipeline model
model = joblib.load("salary_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Column order expected by model
model_columns = [
    'age', 'workclass', 'educational-num', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'hours-per-week', 'native-country',
    'capital-gain-log', 'has-capital-gain', 'has-capital-loss',
    'capital-loss-log', 'hours-per-week-scaled'
]

# App title
st.title("Salary Prediction Application")
st.subheader("Will your income be >50K? Let's find out!")

# Name
name = st.text_input("Enter your name", value="User")

# User Inputs - Original Features
age = st.slider("Age", 17, 90, 30)
education_num = st.selectbox("Education Level (Num)", list(range(1, 17)))
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

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

# Predict button
if st.button("Predict Income"):
    # Build input DataFrame from original values
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

    # Applying the Label Encoding.
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        raw_input[col] = encoders[col].transform(raw_input[col])

    # === Derived Features ===
    raw_input["capital-gain-log"] = np.log1p(raw_input["capital-gain"])
    raw_input["has-capital-gain"] = raw_input["capital-gain"].apply(lambda x: 1 if x > 0 else 0)

    raw_input["capital-loss-log"] = np.log1p(raw_input["capital-loss"])
    raw_input["has-capital-loss"] = raw_input["capital-loss"].apply(lambda x: 1 if x > 0 else 0)

    # Match scaling logic used in training
    raw_input["hours-per-week-scaled"] = (raw_input["hours-per-week"] - 40) / 40  # Modify if you used different scaling

    # Drop unused original features
    raw_input.drop(columns=["capital-gain", "capital-loss"], inplace=True)

    # Reorder columns as expected by the model
    input_data = raw_input[model_columns]

    # Prediction
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Output
    st.markdown(f"### Hai **{name}**,")
    if pred == ">50K":
        st.success(f"Predicted Income: **>50K**")
        st.markdown(f"Confidence: **{prob:.2%}**")
    else:
        st.info(f"Predicted Income: **<=50K**")
        st.markdown(f"Confidence: **{(1 - prob):.2%}**")
