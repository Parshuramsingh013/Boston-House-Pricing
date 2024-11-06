import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Streamlit app title
st.title("Boston House Price Prediction")

# Input fields for all 13 features
crim = st.number_input("Per capita crime rate by town (CRIM)")
zn = st.number_input("Proportion of residential land zoned for lots over 25,000 sq. ft. (ZN)")
indus = st.number_input("Proportion of non-retail business acres per town (INDUS)")
chas = st.selectbox("Charles River dummy variable (CHAS) (1 if bounds river; 0 otherwise)", [0, 1])
nox = st.number_input("Nitric oxide concentration (parts per 10 million) (NOX)")
rm = st.number_input("Average number of rooms per dwelling (RM)")
age = st.number_input("Proportion of owner-occupied units built prior to 1940 (AGE)")
dis = st.number_input("Weighted distances to five Boston employment centers (DIS)")
rad = st.number_input("Index of accessibility to radial highways (RAD)")
tax = st.number_input("Full-value property tax rate per $10,000 (TAX)")
ptratio = st.number_input("Pupil-teacher ratio by town (PTRATIO)")
b = st.number_input("1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town (B)")
lstat = st.number_input("Percentage of lower-status population (LSTAT)")

# Prediction button
if st.button("Predict"):
    # Gather all features in an array
    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)
    
    # Predict using the model
    prediction = model.predict(scaled_features)
    
    # Display the prediction
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")
