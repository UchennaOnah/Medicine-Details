import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained models
model = joblib.load('gradient_boosting_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load saved encoders
label_encoder_medicine_name = joblib.load('label_encoder_medicine_name.joblib')
label_encoder_uses = joblib.load('label_encoder_uses.joblib')
label_encoder_composition = joblib.load('label_encoder_composition.joblib')
label_encoder_manufacturer = joblib.load('label_encoder_manufacturer.joblib')
label_encoder_side_effects = joblib.load('label_encoder_side_effects.joblib')
label_encoder_image_url = joblib.load('label_encoder_image_url.joblib')

# Title and description
st.title('Medicine Review Score Prediction')
st.write(''' This application predicts patients review score rating based on salt composition and related features. 
Choose the model you want to use for predictions. 
''')

# Sidebar for user input
st.sidebar.header('Input Medicine Details')


# Define input fields
def user_input_features():
    Medicine_Name = st.sidebar.text_input('Medicine Name (e.g., Avastin 400mg Injection)')
    Composition = st.sidebar.text_input('Composition (e.g., paracetamol)')
    Uses = st.sidebar.text_input('Uses (e.g., pain relief)')
    Side_Effects = st.sidebar.text_input('Side_effects (e.g., Nausea)')
    Manufacturer = st.sidebar.text_input('Manufacturer (e.g., ABC Pharma)')
    Excellent_Review = st.sidebar.slider('Number of Excellent Reviews', 0, 500, 10)
    Average_Review = st.sidebar.slider('Number of Average Reviews', 0, 500, 5)
    Poor_Review = st.sidebar.slider('Number of Poor Reviews', 0, 500, 1)

    # Validate input
    if not Medicine_Name or not Composition or not Uses or not Manufacturer:
        st.error('Please provide valid input for all fields!')

    # Create a dataframe for input features
    data = {
        'Medicine Name': [Medicine_Name],
        'Composition': [Composition],
        'Uses': [Uses],
        'Side_effects': [Side_Effects],
        'Manufacturer': [Manufacturer],
        'Excellent Review %': [Excellent_Review],
        'Average Review %': [Average_Review],
        'Poor Review %': [Poor_Review]

    }
    return pd.DataFrame(data, index=[0]), Excellent_Review, Average_Review, Poor_Review


# Get user input
input_data, Excellent_Review, Average_Review, Poor_Review = user_input_features()

# Display input
st.subheader('User Input:')
st.write(input_data)

if input_data is not None:
    input_data['Medicine Name'] = label_encoder_medicine_name.fit_transform(input_data['Medicine Name'])
    input_data['Uses'] = label_encoder_uses.fit_transform(input_data['Uses'])
    input_data['Composition'] = label_encoder_composition.fit_transform(input_data['Composition'])
    input_data['Manufacturer'] = label_encoder_manufacturer.fit_transform(input_data['Manufacturer'])

    # Ensure column names and order match the training data
    expected_columns = ['Medicine Name', 'Composition', 'Uses', 'Side_effects', 'Manufacturer',
                        'Excellent Review %', 'Average Review %', 'Poor Review %']
    input_data = input_data[expected_columns]

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    if st.button('Predict Review Score'):
        prediction = model.predict(scaled_data)
        st.subheader('Predicted Review Score:')
        st.write(prediction[0])



# Style the app
st.markdown('### Welcome to the Medicine Review Predictor!')

import matplotlib.pyplot as plt

# Visualize review distribution
fig, ax = plt.subplots()
ax.bar(['Excellent Review %', 'Average Review %', 'Poor Review %'], [Excellent_Review, Average_Review, Poor_Review])
st.pyplot(fig)

