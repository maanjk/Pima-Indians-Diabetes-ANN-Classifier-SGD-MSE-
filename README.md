# Pima Indians Diabetes ANN Classifier (Streamlit App)

Artificial Neural Network (ANN) classifier for the **Pima Indians Diabetes** dataset  
from Kaggle, trained with **Stochastic Gradient Descent (SGD)** and deployed as a  
Streamlit web application.

> **Assignment constraints:**  
> - Learning Rate: `0.01`  
> - Batch Size: `16`  
> - Epochs: `50`  
> - Loss Function: **Mean Squared Error (MSE)**  
> - Optimizer: **Stochastic Gradient Descent (SGD)**  
> - Model type: Feed‑forward ANN (Dense layers only)

---

## 1. Dataset

- **Name:** Pima Indians Diabetes Database  
- **Source:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
- **Task:** Binary classification (Outcome: `0` = non‑diabetic, `1` = diabetic)  
- **Features:** 8 numeric medical attributes (Pregnancies, Glucose, BMI, Age, etc.)

The raw CSV file (`diabetes.csv`) is *not* stored in this repository due to Kaggle’s
terms of use. It can be downloaded directly from Kaggle.

---

## 2. Model & Training

### 2.1 Pre‑processing

- Treated biologically impossible zeros as missing values in:
  - `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`
- Imputed missing values with **median** (`sklearn.impute.SimpleImputer`)
- Standardized all features with **StandardScaler** (`sklearn.preprocessing.StandardScaler`)
- Train/test split: 80% / 20%, stratified by target

### 2.2 ANN Architecture

Implemented using **TensorFlow / Keras** (`Sequential` API):

```text
Input (8 features)
↓
Dense(32, activation="relu", kernel_initializer="he_normal", L2 regularization)
Dropout(0.3)
↓
Dense(16, activation="relu", kernel_initializer="he_normal", L2 regularization)
Dropout(0.2)
↓
Dense(1, activation="sigmoid")  # Binary output
