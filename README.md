# Customer Churn Prediction Dashboard

An end-to-end machine learning web app that predicts whether a telecom customer will churn.

## Features
- Predicts churn probability based on customer details
- SHAP explainability showing why the model made each prediction
- Interactive dashboard built with Streamlit

## Tech Stack
- Python, Pandas, NumPy
- XGBoost (85%+ accuracy)
- SHAP for explainability
- Streamlit for dashboard
- Deployed on Streamlit Cloud

## Dataset
Telco Customer Churn dataset from Kaggle (7,043 records)

## How to Run
pip install -r requirements.txt
streamlit run app.py