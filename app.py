
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# Create input fields
credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=850, value=600
)
age = st.number_input("Age", min_value=0, max_value=120, value=40)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input(
    "Number of Products", min_value=1, max_value=4, value=2
)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1], index=1)
is_active_member = st.selectbox("Is Active Member", options=[0, 1], index=1)
estimated_salary = st.number_input(
    "Estimated Salary", min_value=0.0, value=50000.0
)

geography = st.selectbox(
    "Geography", options=['France', 'Spain', 'Germany']
)
gender = st.selectbox("Gender", options=['Male', 'Female'])

# Encode gender using the fitted encoder
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform(
    pd.DataFrame({'Geography': [geography]})
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Prepare input data, including Gender
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine numerical, encoded gender, and geography features
input_data = pd.concat(
    [input_data, geo_encoded_df],
    axis=1
)

# Match the exact feature names and order used during training
input_data = input_data.reindex(
    columns=scaler.feature_names_in_
)

# Scale the input data
scaled_input_data = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_input_data, verbose=0)
prediction_proba = float(prediction[0][0])

st.subheader("Prediction Results")
st.metric("Churn Probability", f"{prediction_proba:.2%}")

if prediction_proba > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is unlikely to churn.")
