import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
try:
    model = joblib.load("ensemble_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.write(f"Error loading model or vectorizer: {e}")

# Title and description
st.title("Fake News Detector")
st.write("Enter a news headline or article text to check if it is likely real or fake.")

# Input field for the user to enter news text
user_input = st.text_area("Enter news text here:", "")

# Convert `user_input` to lowercase string and wrap in list
user_input_reshaped = [str(user_input).lower()]  # Ensure it's a list of one string

# Button to trigger prediction
if st.button("Detect"):
    if user_input:  # Ensure there's input text
        try:
            # Vectorize input (transform to 2D array for model)
            user_input_vectorized = vectorizer.transform(user_input_reshaped)

            # Make prediction
            prob = model.predict_proba(user_input_vectorized)

            # Display prediction probabilities (ensure the model is working)
            fake_prob = prob[0][0] * 100  # Probability of being fake in percentage
            real_prob = prob[0][1] * 100  # Probability of being real in percentage

            # Display result
            st.write(f"🛑 Probability of Fake News: {fake_prob:.2f}%")
            st.write(f"✅ Probability of Real News: {real_prob:.2f}%")
            if fake_prob > real_prob:
                st.write(f"🚨 The news is **likely fake** with a probability of {fake_prob:.2f}%")
            else:
                st.write(f"✅ The news is **likely real** with a probability of {real_prob:.2f}%")
        except Exception as e:
            st.error(f"Error during vectorization or prediction: {e}")
    else:
        st.warning("Please enter some text to check.")
