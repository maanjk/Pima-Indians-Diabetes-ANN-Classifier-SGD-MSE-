import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model_and_tools():
    model_path = BASE_DIR / "models" / "diabetes_ann.h5"
    scaler_path = BASE_DIR / "models" / "scaler.pkl"
    imputer_path = BASE_DIR / "models" / "imputer.pkl"

    # Load model WITHOUT compiling (avoids loss/metric deserialization issues)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Optional: recompile if you want, but not strictly required for predict()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="mse",
        metrics=["accuracy"],
    )

    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    return model, scaler, imputer

model, scaler, imputer = load_model_and_tools()
