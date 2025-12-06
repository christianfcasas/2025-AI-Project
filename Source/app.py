import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from PIL import Image
model_path = os.path.join(os.path.dirname(__file__), 'random_forest_model.joblib')
model = joblib.load(model_path)
logo_path = os.path.join(os.path.dirname(__file__), 'logo and name.png')
logo = Image.open(logo_path)
# Define the feature columns
feature_columns = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other', #'ever_married_Yes',
    'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
    'work_type_children', 'Residence_type_Urban',
    'smoking_status_formerly smoked', 'smoking_status_never smoked',
    'smoking_status_smokes'
]

# Streamlit app
st.image(logo, width=150) 
#st.title("Stroke Prediction App")
st.write("Enter your personal data to predict your stroke status.")

age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
bmi = st.number_input("BMI", min_value=0.0)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", ["","Yes", "No"])
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", ["","Yes", "No"])
work_type = st.selectbox("Work Type", ["","Private", "Self-employed", "Govt_job", "Never_worked", "No - Children"])
residence_type = st.selectbox("Residence Type", ["", "Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["","never smoked", "formerly smoked", "smokes", "Unknown"])
ever_married = st.selectbox("Ever Married", ["","Yes", "No"])


# Convert user input into a DataFrame
user_data = pd.DataFrame({
    'age': [age],
    'hypertension': [1 if hypertension == "Yes" else 0],
    'heart_disease': [1 if heart_disease == "Yes" else 0],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'gender_Male': [1 if gender == "Male" else 0],
    'gender_Other': [1 if gender == "Other" else 0],
    #'ever_married_Yes': [1 if ever_married == "Yes" else 0],
    'work_type_Never_worked': [1 if work_type == "Never_worked" else 0],
    'work_type_Private': [1 if work_type == "Private" else 0],
    'work_type_Self-employed': [1 if work_type == "Self-employed" else 0],
    'work_type_children': [1 if work_type == "No - Children" else 0],
    'Residence_type_Urban': [1 if residence_type == "Urban" else 0],
    'smoking_status_formerly smoked': [1 if smoking_status == "formerly smoked" else 0],
    'smoking_status_never smoked': [1 if smoking_status == "never smoked" else 0],
    'smoking_status_smokes': [1 if smoking_status == "smokes" else 0]
})

# Ensure the columns match the model's expected input
user_data = user_data[feature_columns]

# Predict stroke status
if st.button("Predict"):
    if "" in [gender, hypertension, heart_disease, work_type, ever_married, residence_type, smoking_status]:
            st.error("Please complete all fields before predicting.")
    else:
        if 0 in [age, avg_glucose_level, bmi]:
            st.error("Please complete all fields before predicting.")
        else:
            threshold = 0.2
            prediction = model.predict(user_data)[0]
            probability = round(model.predict_proba(user_data)[0][1], 4)
            prediction = 1 if probability > threshold else 0
            st.write(f"**Threshold:** {threshold*100:.0f}%")
            if prediction==1:
                st.error(f"The model predicts that you have a **Higher** Stroke Risk with a probability of **{probability*100:.2f}%**.")
            else:
                st.success(f"The model predicts that you have a **Lower** Stroke Risk with a probability of **{probability*100:.2f}%**.")