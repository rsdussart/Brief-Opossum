import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image

# Load models
lin_reg = pickle.load(open('linear.pkl', 'rb'))
ridge = pickle.load(open('ridge.pkl', 'rb'))
lasso = pickle.load(open('lasso.pkl', 'rb'))
data_opossum = pickle.load(open('data_opossum.pkl', 'rb'))

# Define the user interface
st.title("Opossum Age Predictor")

model_selection = st.selectbox("Select a model", ["Linear Regression", "Ridge Regression", "Lasso Regression"])

hdlngth = st.number_input("Hdlngth", min_value=0, max_value=1000, value=0)
skullw = st.number_input("Skullw", min_value=0, max_value=1000, value=0)
totlngth = st.number_input("Totlngth", min_value=0, max_value=1000, value=0)
taill = st.number_input("Taill", min_value=0, max_value=1000, value=0)
footlgth = st.number_input("Footlgth", min_value=0, max_value=1000, value=0)
earconch = st.number_input("Earconch", min_value=0, max_value=1000, value=0)
eye = st.number_input("Eye", min_value=0, max_value=1000, value=0)
chest = st.number_input("Chest", min_value=0, max_value=1000, value=0)
belly = st.number_input("Belly", min_value=0, max_value=1000, value=0)

poly = PolynomialFeatures()
poly.fit(data_opossum)
input_data_opossum= poly.transform(data_opossum)

if st.button("Predict"):
    input_data = [hdlngth, skullw, totlngth, taill, footlgth, earconch, eye, chest, belly]
    input_data = np.array(input_data).reshape(1, -1)
    input_data = poly.fit_transform(input_data)

    if model_selection == "Linear Regression":
        prediction = lin_reg.predict(input_data)
    elif model_selection == "Ridge Regression":
        prediction = ridge.predict(input_data)
    else:
        prediction = lasso.predict(input_data)

    opossum_age = prediction[0]
    age_range = "Baby" if opossum_age < 1 else "Young" if opossum_age < 3 else "Adult"

    if age_range == "Baby":
        image = Image.open("baby.jpg")
    elif age_range == "Young":
        image = Image.open("young.jpg")
    else:
        image = Image.open("adulte.jpg")

    st.image(image, caption="Predicted age range: " + age_range, use_column_width=True)
    st.write("Predicted age:", opossum_age)
    st.write("Age range:", age_range)
    st.image(image, caption="Predicted age range: " + age_range, use_column_width=True)





