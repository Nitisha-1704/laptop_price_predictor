# Import required libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the trained pipeline model
model = joblib.load("pipe.pkl")  # This includes preprocessing + regression model

# Streamlit UI
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Predictor")
st.write("Enter the specifications of your desired laptop to estimate its price.")

# Input fields
Company_input = st.text_input("Company", value="HP")
Typename = st.text_input("Type Name", value='Notebook')
Ram_input = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8)
Weight_input = st.number_input("Weight (Kg)", min_value=0.69, max_value=4.7, value=1.0)
Touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
Ips_input = st.selectbox("IPS Display", ["No", "Yes"])
ppi_input = st.number_input("PPI (Pixels Per Inch)", min_value=90.58, max_value=352.46, value=120.0)
cpu_brand = st.text_input("CPU Brand", value="Intel Core i5")
HDD = st.number_input("HDD (in GB)", min_value=0, max_value=1000, value=0)
SSD = st.number_input("SSD (in GB)", min_value=0, max_value=2000, value=0)

Gpu_brand = st.text_input("GPU Brand", value='Intel')
OS = st.text_input("Operating System", value='Windows')

# Handle categorical inputs
Touchscreen = 1 if Touchscreen == "Yes" else 0
Ips_input = 1 if Ips_input == "Yes" else 0

# Predict button
if st.button("Predict Price"):
    try:
        # Prepare input dictionary
        input_dict = {
            'Company': [Company_input],
            'TypeName': [Typename],
            'Ram': [Ram_input],
            'Weight': [Weight_input],
            'Touchscreen': [Touchscreen],
            'Ips': [Ips_input],
            'ppi': [ppi_input],
            'cpu_brand': [cpu_brand],
            'HDD': [int(HDD)],
            'SSD': [int(SSD)],
            'Gpu_brand': [Gpu_brand],
            'os': [OS]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_dict)

        # Directly predict using the pipeline
        prediction = model.predict(input_df)[0]

        # Display result
        st.subheader("💰 Estimated Laptop Price:")
        st.success(f"₹ {round(prediction, 2)}")

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
