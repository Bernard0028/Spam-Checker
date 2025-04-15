import streamlit as st
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📱 Spam SMS Classifier")

message = st.text_area("Enter your SMS message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vector = vectorizer.transform([message])
        prediction = model.predict(input_vector)[0]

        if prediction == "spam" or prediction == 1:
            st.error("❗ This is likely a SPAM message.")
        else:
            st.success("✅ This message seems to be HAM (not spam).")
