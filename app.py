import streamlit as st
import pickle
from preprocess import clean_text

# Load model
with open("model.sav", "rb") as f:
    model = pickle.load(f)

st.title("Mail Classification App")

user_input = st.text_area("Enter Mail to classify")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    prediction = model.predict([user_input])[0]
    st.success(f"Prediction is : {prediction}")
