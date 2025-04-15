import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("📩 SMS Spam Classifier")
st.write("Enter a text message and the app will classify it as **Spam** or **Ham (Not Spam)**.")

# Text input
user_input = st.text_area("📨 Your Message", height=150)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Transform input
        input_tfidf = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_tfidf)[0]
        
        # Display result
        if prediction == 1:
            st.error("🚫 This message is predicted to be **Spam**.")
        else:
            st.success("✅ This message is predicted to be **Ham (Not Spam)**.")
