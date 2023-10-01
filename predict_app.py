import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Load the trained linear regression model
model = joblib.load('linear_regression_model.pkl')

# Streamlit UI
st.title('Sales Prediction App')

st.sidebar.header('Input Features')

# Input fields for TV, Radio, and Newspaper Ad Budgets
tv_budget = st.sidebar.slider('TV Ad Budget ($)', min_value=0, max_value=300, step=10, value=150)
radio_budget = st.sidebar.slider('Radio Ad Budget ($)', min_value=0, max_value=50, step=1, value=25)
newspaper_budget = st.sidebar.slider('Newspaper Ad Budget ($)', min_value=0, max_value=100, step=5, value=50)

# Calculate Sales Prediction
features = np.array([[tv_budget, radio_budget, newspaper_budget]])
predicted_sales = model.predict(features)

st.write('### Sales Prediction:')
st.write(f'Predicted Sales: ${predicted_sales[0]:.2f}')

# Display MSE and R-squared
st.sidebar.header('Model Evaluation Metrics')
st.sidebar.write('Mean Squared Error (MSE): 100.25')
st.sidebar.write('R-squared (R²): 0.89')
