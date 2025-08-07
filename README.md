# 🔁 Customer Churn Prediction using Artificial Neural Network

This project predicts whether a customer will churn (leave the service) or stay, using a trained Artificial Neural Network (ANN) model. The dataset includes features such as credit score, geography, age, balance, number of products, and more. A Streamlit web application is used for user-friendly predictions.

---

## 📌 Project Objective

The main goal is to develop a machine learning model that can classify whether a customer will churn based on given attributes, and deploy it using Streamlit for real-time prediction.

---

## 🧠 Technologies & Tools Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
- Streamlit (for UI)
- Git & GitHub

---

## 📁 Project Structure
churn-modeling/

├── app.py # Streamlit web app

├── model.pkl # Trained ANN model

├── requirements.txt # Required Python libraries

├── churn_data.csv # Dataset for training

├── churn_model.ipynb # Jupyter notebook (model building)

└── README.md # Project documentation

📊 Dataset Overview
File: churn_data.csv

Target: Exited (1 = Churned, 0 = Stayed)

Features:

CreditScore

Geography

Gender

Age

Tenure

Balance

NumOfProducts

HasCrCard

IsActiveMember

EstimatedSalary
