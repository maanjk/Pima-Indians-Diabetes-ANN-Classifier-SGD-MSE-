import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and preprocessors once
@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model("models/diabetes_ann.h5")
    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")
    return model, scaler, imputer

model, scaler, imputer = load_model_and_tools()

st.title("Pima Indians Diabetes Prediction – ANN")
st.write(
    "This app uses an Artificial Neural Network (ANN) trained on the "
    "Kaggle **Pima Indians Diabetes** dataset.\n\n"
    "Training setup: SGD optimizer, learning rate 0.01, batch size 16, "
    "50 epochs, loss = Mean Squared Error."
)

st.sidebar.header("Input Patient Data")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=250, value=120, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=79, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=33, step=1)

if st.button("Predict"):
    # Arrange features in the same order as training data
    x = np.array([[pregnancies, glucose, blood_pressure,
                   skin_thickness, insulin, bmi, dpf, age]],
                 dtype=np.float32)

    # Apply imputer (for any missing values) and scaler
    x_imputed = imputer.transform(x)
    x_scaled = scaler.transform(x_imputed)

    # Predict
    prob = float(model.predict(x_scaled)[0][0])
    label = "Diabetic" if prob >= 0.5 else "Non-diabetic"

    st.subheader("Prediction")
    st.write(f"Estimated probability of diabetes: **{prob:.3f}**")
    st.write(f"Predicted class: **{label}**")
    st.caption("Note: Educational demo only, not for real medical use.")