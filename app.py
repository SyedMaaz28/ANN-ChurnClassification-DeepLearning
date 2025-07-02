import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder , StandardScaler
import pickle
from keras.models import load_model


## Load the Trained Model
model =load_model('model.h5')

## Load the Scaler and Encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

## Streamlit App
st.title("Customer Churn Prediction")

# Input fields for user data
geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", 18, 90)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (in years)", 0, 10)
number_of_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]      
})


## One hot encode the geography
geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))


# Concatenate the geography encoded data with the input data
input_df = pd.concat([input_data.reset_index(drop = True), geography_encoded_df], axis=1)

# Scale the input data
input_df_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_df_scaled)      
prediction_proba = prediction[0][0]

# Display the prediction result
if prediction_proba > 0.5:
    st.write("The customer is likely to churn with a probability of → {:.2f}%.".format(prediction_proba * 100))
else:
    st.write("The customer is not likely to churn with a probability of →  {:.2f}%.".format((1 - prediction_proba) * 100))
