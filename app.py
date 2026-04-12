import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Models (Cached)
# -----------------------------
@st.cache_resource
def load_model_files():
    model = tf.keras.models.load_model('model.h5')

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_files()

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction (ANN)")
st.markdown("Predict whether a customer is likely to churn using a trained Artificial Neural Network")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Details")

geography = st.sidebar.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider("Age", 18, 92, 35)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)

num_of_products = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    2
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.sidebar.selectbox(
    "Is Active Member",
    [0, 1]
)

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Scale input
input_data_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Churn"):

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.subheader("Prediction Result")

    # Probability bar
    st.progress(float(prediction_proba))

    st.metric(
        label="Churn Probability",
        value=f"{prediction_proba:.2%}"
    )

    if prediction_proba > 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is NOT likely to churn")

# -----------------------------
# Show Input Data
# -----------------------------
with st.expander("View Input Data"):
    st.write(input_data)