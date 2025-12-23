import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("fraud_xgb_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's **Fraudulent** or **Normal**")

# User Inputs
category = st.selectbox(
    "Transaction Category",
    [
        "gas_transport", "grocery_pos", "home", "shopping_pos",
        "kids_pets", "shopping_net", "entertainment", "food_dining",
        "personal_care", "health_fitness", "misc_pos", "misc_net",
        "grocery_net", "travel"
    ]
)

gender = st.selectbox("Gender", ["M", "F"])

state = st.text_input("State Code (e.g., CA, NY)", "CA")

zip_code = st.number_input("ZIP Code", min_value=10000, max_value=99999, value=90001)

lat = st.number_input("Customer Latitude", value=34.05)
long = st.number_input("Customer Longitude", value=-118.24)

city_pop = st.number_input("City Population", min_value=0, value=50000)

merch_lat = st.number_input("Merchant Latitude", value=34.05)
merch_long = st.number_input("Merchant Longitude", value=-118.25)

amount = st.number_input("Transaction Amount ($)", min_value=1.0, value=120.0)

hour = st.slider("Transaction Hour", 0, 23, 14)
dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

is_weekend = 1 if dayofweek in [5, 6] else 0

# Feature engineering
log_amt = np.log1p(amount)

R = 6371
lat1 = np.radians(lat)
lon1 = np.radians(long)
lat2 = np.radians(merch_lat)
lon2 = np.radians(merch_long)

dlat = lat2 - lat1
dlon = lon2 - lon1

a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
cust_merch_dist = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Input DataFrame
input_data = pd.DataFrame({
    "category": [category],
    "gender": [gender],
    "state": [state],
    "zip": [zip_code],
    "lat": [lat],
    "long": [long],
    "city_pop": [city_pop],
    "merch_lat": [merch_lat],
    "merch_long": [merch_long],
    "log_amt": [log_amt],
    "hour": [hour],
    "dayofweek": [dayofweek],
    "is_weekend": [is_weekend],
    "cust_merch_dist": [cust_merch_dist]
})

if st.button("🔍 Predict Fraud"):
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Result")
    st.write(f"Fraud Probability: **{prob:.2%}**")

    if prob > 0.3:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Normal Transaction")
