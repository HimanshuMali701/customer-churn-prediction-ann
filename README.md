# Customer Churn Prediction (ANN + Streamlit)

Predict whether a bank customer is likely to **churn** using an **Artificial Neural Network (TensorFlow/Keras)** and deploy it with **Streamlit**.

---

## 🚀 Live Demo  
[![Open App](https://img.shields.io/badge/Launch-App-blue?style=for-the-badge)](https://huggingface.co/spaces/HimanshuMali/CustomerChurn_Prediction_ANN)

_(Ctrl + Click to open in new tab)_
This project takes customer details and predicts:

* Churn Probability
* Churn / Not Churn decision

---

## Features

* ANN Deep Learning model (TensorFlow/Keras)
* Streamlit interactive UI
* OneHotEncoding + LabelEncoding
* StandardScaler preprocessing
* Real-time prediction
* Probability score visualization
* Clean deployment-ready structure

---

## Tech Stack
- TensorFlow / Keras (ANN)
- Python
- Scikit-learn (Preprocessing)
- Pandas / NumPy
- Streamlit (UI)
- Docker (Deployment)
- Hugging Face Spaces
---
## Key Highlights
- Built ANN model for churn prediction
- Applied feature engineering and encoding
- Deployed using Docker on Hugging Face
- Real-time prediction with probability output
---
## Project Structure

```
customer-churn-prediction-ann
│
├── app.py
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
├── Churn_Modelling.csv
└── README.md
```

---

## Input Features

The model uses:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

---

## Installation

### 1. Clone repository

```
git clone https://github.com/HimanshuMali701/customer-churn-prediction-ann.git
cd customer-churn-prediction-ann
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run Streamlit app

```
streamlit run app.py
```

---

## Model Architecture

* Input Layer
* Dense Layer (ReLU)
* Dense Layer (ReLU)
* Output Layer (Sigmoid)

Loss Function: Binary Crossentropy
Optimizer: Adam
Task: Binary Classification

---

## Example Output

```
Churn Probability: 0.82
Customer is likely to churn
```

---

## Dataset

Bank Customer Churn Dataset
Contains **10,000 customers** with demographic and account information.

Target:

```
Exited
0 = Not churn
1 = Churn
```

---

## Future Improvements

* SHAP Explainability
* Model comparison (ANN vs XGBoost)
* Batch prediction upload
* Deployment on Streamlit Cloud
* Docker support
---
## Author

Himanshu Mali
