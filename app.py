import streamlit as st
import numpy as np
import joblib

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction System")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0,1])
cp = st.number_input("Chest Pain Type (0-3)", 0,3)
trestbps = st.number_input("Resting Blood Pressure", 80,200)
chol = st.number_input("Cholesterol", 100,600)
fbs = st.selectbox("Fasting Blood Sugar >120 (1=Yes, 0=No)", [0,1])
restecg = st.number_input("Rest ECG (0-2)", 0,2)
thalach = st.number_input("Max Heart Rate", 60,220)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0,1])
oldpeak = st.number_input("ST Depression", 0.0,10.0)
slope = st.number_input("Slope (0-2)", 0,2)
ca = st.number_input("Number of Major Vessels (0-3)", 0,3)
thal = st.number_input("Thal (0-3)", 0,3)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])
    
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
