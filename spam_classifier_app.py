import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📩 Spam SMS Classifier")
st.write("Enter an SMS message below to check if it's spam or not.")

user_input = st.text_area("Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(input_transformed)[0]
        label = "📬 Ham (Not Spam)" if prediction == "ham" or prediction == 0 else "🚫 Spam"
        st.success(f"Prediction: **{label}**")
