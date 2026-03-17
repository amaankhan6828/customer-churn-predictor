import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt

with open('churn_model (1).pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_columns (1).json', 'r') as f:
    feature_columns = json.load(f)

st.title("Customer Churn Predictor")
st.write("Fill in the customer details below to predict if they will churn.")

st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
    total_charges = st.slider("Total Charges ($)", 0, 9000, 1000)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

col4, col5 = st.columns(2)

with col4:
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

with col5:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

if st.button("Predict Churn"):

    input_dict = {col: 0 for col in feature_columns}
    input_dict['gender'] = 1 if gender == 'Male' else 0
    input_dict['SeniorCitizen'] = 1 if senior_citizen == 'Yes' else 0
    input_dict['Partner'] = 1 if partner == 'Yes' else 0
    input_dict['Dependents'] = 1 if dependents == 'Yes' else 0
    input_dict['tenure'] = tenure
    input_dict['PhoneService'] = 1 if phone_service == 'Yes' else 0
    input_dict['MultipleLines'] = 1 if multiple_lines == 'Yes' else 0
    input_dict['OnlineSecurity'] = 1 if online_security == 'Yes' else 0
    input_dict['OnlineBackup'] = 1 if online_backup == 'Yes' else 0
    input_dict['DeviceProtection'] = 1 if device_protection == 'Yes' else 0
    input_dict['TechSupport'] = 1 if tech_support == 'Yes' else 0
    input_dict['StreamingTV'] = 1 if streaming_tv == 'Yes' else 0
    input_dict['StreamingMovies'] = 1 if streaming_movies == 'Yes' else 0
    input_dict['PaperlessBilling'] = 1 if paperless_billing == 'Yes' else 0
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges
    input_dict['InternetService_DSL'] = 1 if internet_service == 'DSL' else 0
    input_dict['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0
    input_dict['InternetService_No'] = 1 if internet_service == 'No' else 0
    input_dict['Contract_Month-to-month'] = 1 if contract == 'Month-to-month' else 0
    input_dict['Contract_One year'] = 1 if contract == 'One year' else 0
    input_dict['Contract_Two year'] = 1 if contract == 'Two year' else 0
    input_dict['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method == 'Bank transfer (automatic)' else 0
    input_dict['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == 'Credit card (automatic)' else 0
    input_dict['PaymentMethod_Electronic check'] = 1 if payment_method == 'Electronic check' else 0
    input_dict['PaymentMethod_Mailed check'] = 1 if payment_method == 'Mailed check' else 0

    input_df = pd.DataFrame([input_dict])[feature_columns]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prob >= 0.5:
        st.error(f"This customer is likely to churn ({prob*100:.1f}% probability)")
    else:
        st.success(f"This customer is unlikely to churn ({prob*100:.1f}% probability)")

    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=feature_columns
    ), show=False)
    st.pyplot(fig)