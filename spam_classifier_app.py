import streamlit as st
import joblib

# Load the tuned model and vectorizer
nb_model_tuned = joblib.load("nb_model_tuned.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📩 Spam SMS Classifier")
st.write("Enter an SMS message below to check if it's spam or not.")

user_input = st.text_area("Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform the input using the vectorizer
        input_transformed = vectorizer.transform([user_input])
        
        # Make a prediction using the tuned model
        prediction = nb_model_tuned.predict(input_transformed)[0]
        
        # Determine the label
        label = "📬 Ham (Not Spam)" if prediction == "ham" else "🚫 Spam"
        
        # Display the result
        st.success(f"Prediction: **{label}**")
