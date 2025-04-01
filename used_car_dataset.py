import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    loaded_model = joblib.load('used_car_price_prediction.joblib')
except FileNotFoundError:
    st.error("Model file 'used_car_price_prediction.joblib' not found. Please ensure the model is saved correctly.")
    st.stop()

# Load the original DataFrame (with encoding and debugging)
try:
    df = pd.read_csv('used_car_dataset.csv', encoding='utf-8')  # Try different encodings if needed
    st.write(df.head())  # Show sample data
    st.write(f"Data type of 'Year': {df['Year'].dtype}")  # Show data type
    st.write(f"Missing values in 'Year': {df['Year'].isnull().sum()}")  # Check for NaN
except FileNotFoundError:
    st.error("Data file 'used_car_dataset.csv' not found. Please ensure your data file is in the same directory.")
    st.stop()

# Handle missing data in relevant columns
df['kmDriven'] = pd.to_numeric(df['kmDriven'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Remove rows with NaN values in critical columns
df = df.dropna(subset=['kmDriven', 'Year', 'Age'])

# Ensure all values are valid for predictions (No NaN, no invalid data types)
assert df['Year'].isnull().sum() == 0, "There are still NaN values in 'Year'"
assert df['kmDriven'].isnull().sum() == 0, "There are still NaN values in 'kmDriven'"
assert df['Age'].isnull().sum() == 0, "There are still NaN values in 'Age'"

st.title('Used Car Price Prediction')

# Input fields for car features
# Handling cases where mean or min/max values could be NaN
year_min = int(df['Year'].min()) if not np.isnan(df['Year'].min()) else 2000  # Set a fallback if NaN
year_max = int(df['Year'].max()) if not np.isnan(df['Year'].max()) else 2025  # Set a fallback if NaN
year_mean = int(df['Year'].mean()) if not np.isnan(df['Year'].mean()) else 2015  # Set a fallback if NaN

age_min = int(df['Age'].min()) if not np.isnan(df['Age'].min()) else 0
age_max = int(df['Age'].max()) if not np.isnan(df['Age'].max()) else 100
age_mean = int(df['Age'].mean()) if not np.isnan(df['Age'].mean()) else 5

km_driven_min = float(df['kmDriven'].min()) if not np.isnan(df['kmDriven'].min()) else 0
km_driven_max = float(df['kmDriven'].max()) if not np.isnan(df['kmDriven'].max()) else 200000
km_driven_mean = float(df['kmDriven'].mean()) if not np.isnan(df['kmDriven'].mean()) else 50000

# Create number input widgets
year = st.number_input('Year', min_value=year_min, max_value=year_max, value=year_mean)
age = st.number_input('Age', min_value=age_min, max_value=age_max, value=age_mean)
km_driven = st.number_input('Kilometers Driven', min_value=km_driven_min, max_value=km_driven_max, value=km_driven_mean)

# Create a DataFrame from the input
input_data = pd.DataFrame([[year, age, km_driven]], columns=['Year', 'Age', 'kmDriven'])

# Predict price on button click
if st.button('Predict Price'):
    try:
        # Prediction
        prediction = loaded_model.predict(input_data)[0]
        st.write(f'Predicted Price: Ksh {prediction:,.2f}')  # Format with commas and 2 decimal places
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
