import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
ridge_model = pickle.load(open('Regression Project/streamlit/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('Regression Project/streamlit/scaler.pkl', 'rb'))

# Set up the Streamlit UI
st.title("Wildfire Burned Area Prediction")

st.write("Fill in the details below to predict the burned area.")

# Input fields
Temperature = st.number_input("Temperature", value=20.0)
RH = st.number_input("Relative Humidity (RH)", value=50.0)
Ws = st.number_input("Wind Speed (Ws)", value=5.0)
Rain = st.number_input("Rain", value=0.0)
FFMC = st.number_input("FFMC Index", value=85.0)
DMC = st.number_input("DMC Index", value=100.0)
ISI = st.number_input("ISI Index", value=10.0)
Classes = st.number_input("Classes (0 or 1)", min_value=0.0, max_value=1.0, value=0.0)
Region = st.number_input("Region (e.g., 1 = Bejaia, 2 = Sidi-Bel)", min_value=1.0, max_value=2.0, value=1.0)

# Predict button
if st.button("Predict Burned Area"):
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    scaled_input = standard_scaler.transform(input_data)
    prediction = ridge_model.predict(scaled_input)
    st.success(f"Predicted Burned Area: {prediction[0]:.2f} ha")
