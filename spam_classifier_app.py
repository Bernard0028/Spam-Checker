import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📱 SMS Spam Classifier")

# User input
message = st.text_area("Enter your SMS message:")

if st.button("Predict"):
    if message.strip():
        # Vectorize the input
        msg_vector = vectorizer.transform([message])
        prediction = model.predict(msg_vector)[0]
        
        # Output
        label = "📩 Ham" if prediction == 0 else "🚨 Spam"
        st.success(f"Prediction: {label}")
    else:
        st.warning("Please enter a message.")
