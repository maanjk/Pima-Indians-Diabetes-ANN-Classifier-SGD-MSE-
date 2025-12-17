import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

# Basic page config
st.set_page_config(page_title="Pima Indians Diabetes ANN", layout="centered")

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model_and_tools():
    model_path = BASE_DIR / "models" / "diabetes_ann.h5"
    scaler_path = BASE_DIR / "models" / "scaler.pkl"
    imputer_path = BASE_DIR / "models" / "imputer.pkl"

    # Load model WITHOUT compiling (compile is only needed for training)
    model = tf.keras.models.load_model(model_path, compile=False)

    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    return model, scaler, imputer


def main():
    st.title("Pima Indians Diabetes Prediction – ANN")
    st.write(
        "This app uses an Artificial Neural Network (ANN) trained on the "
        "Kaggle **Pima Indians Diabetes** dataset.\n\n"
        "Training: SGD optimizer, learning rate 0.01, batch size 16, "
        "50 epochs, loss = MSE."
    )

    # Try to load model; if anything fails, show it on the page
    try:
        model, scaler, imputer = load_model_and_tools()
    except Exception as e:
        st.error("Failed to load model or preprocessing objects.")
        st.exception(e)
        return

    st.sidebar.header("Input Patient Data")

    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1, 1)
    glucose = st.sidebar.number_input("Glucose", 0, 250, 120, 1)
    blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 200, 70, 1)
    skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20, 1)
    insulin = st.sidebar.number_input("Insulin", 0, 900, 79, 1)
    bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 32.0, 0.1)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
    age = st.sidebar.number_input("Age", 10, 100, 33, 1)

    if st.button("Predict"):
        x = np.array([[pregnancies, glucose, blood_pressure,
                       skin_thickness, insulin, bmi, dpf, age]],
                     dtype=np.float32)

        x_imputed = imputer.transform(x)
        x_scaled = scaler.transform(x_imputed)

        prob = float(model.predict(x_scaled)[0][0])
        label = "Diabetic" if prob >= 0.5 else "Non-diabetic"

        st.subheader("Prediction")
        st.write(f"Estimated probability of diabetes: **{prob:.3f}**")
        st.write(f"Predicted class: **{label}**")
        st.caption("Educational demo only, not medical advice.")


if __name__ == "__main__":
    main()
