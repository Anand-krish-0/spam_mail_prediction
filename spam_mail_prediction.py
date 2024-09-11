import streamlit as st
import pickle

def load_data():
    with open("spam_mail.pkl", 'rb') as file:
        data = pickle.load(file)
    return data

data = load_data()
model = data["model"]
feature_extraction = data["vectorizer"]

def predict_page():
    st.title("Spam Mail Prediction")
    x = st.text_area("Enter the mail: ")

    ok = st.button("Predict")

    if ok:
        # Ensure x is a list of strings as some transformers expect it
        input_data_features = feature_extraction.transform([x])

        # Making prediction
        prediction = model.predict(input_data_features)
        st.write(f"Prediction: {prediction}")

        if prediction[0] == 1:
            st.success('Ham mail')
        else:
            st.warning('Spam mail')

predict_page()
