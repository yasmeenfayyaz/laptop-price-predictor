import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page Setup
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.title("💻 Laptop Price Predictor")
st.write("Enter the laptop specifications below to predict the estimated price.")

# 2. Safely Load Model and Dataframe
try:
    df = pickle.load(open('df.pkl', 'rb'))
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except Exception as e:
    st.error(f"⚠️ Could not load model or dataset file. Please make sure 'df.pkl' and 'pipe.pkl' are uploaded to the main GitHub repository folder.\n\nDetails: {e}")
    st.stop()  # Stops execution here so NameError won't occur below

# 3. User Inputs Form
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type_name = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight of Laptop (in kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)

with col2:
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Display', ['No', 'Yes'])
    screen_size = st.number_input('Screen Size (in Inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# 4. Predict Button & Logic
if st.button('Predict Price 🚀'):
    # PPI Calculation
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Convert Binary Features
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Create Query DataFrame (Feature names match model training exact order)
    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    # Model Prediction
    prediction = pipe.predict(query)[0]
    predicted_price = np.exp(prediction)  # Inverse log transform

    st.success(f"💰 Estimated Laptop Price: **${int(predicted_price):,}**")
