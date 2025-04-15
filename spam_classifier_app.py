import streamlit as st
import joblib
joblib.dump(nb_model_tuned, "nb_model_tuned.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load("nb_model_tuned.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("📱 Spam SMS Classifier")
st.write("Enter an SMS message below and the model will classify it as spam or ham.")

# Text input
user_input = st.text_area("Enter your message here:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Transform input
        input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(input_transformed)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **HAM** (Not Spam).")
