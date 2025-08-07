import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("model.h5")  # Trained ANN model
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#1f77b4;">📊 Customer Churn Prediction</h1>
        <h3 style="color:#FF4500;">Presented to <b>Sir Hamza</b></h3>
        <h4 style="color:#555;">Developed by <b>Bushra Sarwar</b></h4>
    </div>
    <hr>
""", unsafe_allow_html=True)

st.write("### 🔍 Enter Customer Details Below:")

# -----------------------------
# Input Fields
# -----------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=1000, value=650)
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
# -----------------------------
# Data Preprocessing
# -----------------------------
gender_map = {"Female": [1], "Male": [0]}
geo_map = {"France": [1, 0, 0], "Spain": [0, 1, 0], "Germany": [0, 0, 1]}
input_data = np.array([
    credit_score,
    *gender_map[gender],
    age,
    tenure,
    balance,
    num_products,
    has_cr_card,
    is_active_member,
    estimated_salary,
    *geo_map[geography],
]).reshape(1, -1)

# Scale numerical data
scaled_data = scaler.transform(input_data)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Churn"):
    prediction = model.predict(scaled_data)[0][0]
    churn = "❌ Customer Will Leave" if prediction > 0.5 else "✅ Customer Will Stay"
    st.success(f"**Prediction:** {churn} | **Confidence:** {prediction:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
    <hr>
    <p style="text-align:center; color:gray;">Powered by Artificial Neural Network (Churn Model)</p>
""", unsafe_allow_html=True)
