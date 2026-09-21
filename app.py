import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# Page configuration
# =========================

st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# =========================
# Load model
# =========================

model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# =========================
# Header
# =========================

st.title("🏠 California Housing Price Predictor")

st.write(
    "Predict the median housing value of a California area "
    "using a Machine Learning model."
)

st.divider()

# =========================
# Input section
# =========================

st.subheader("📊 Enter Housing Information")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "Longitude",
        value=-122.23
    )

    latitude = st.number_input(
        "Latitude",
        value=37.88
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        value=41.0,
        min_value=1.0
    )

    total_rooms = st.number_input(
        "Total Rooms in Area",
        value=880.0,
        min_value=1.0
    )

with col2:
    total_bedrooms = st.number_input(
        "Total Bedrooms in Area",
        value=129.0,
        min_value=1.0
    )

    population = st.number_input(
        "Population in Area",
        value=322.0,
        min_value=1.0
    )

    households = st.number_input(
        "Households in Area",
        value=126.0,
        min_value=1.0
    )

    median_income = st.number_input(
        "Median Income",
        value=8.3,
        min_value=0.0
    )

ocean_proximity = st.selectbox(
    "🌊 Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

st.divider()

# =========================
# Prediction
# =========================

if st.button("🔮 Predict Price", use_container_width=True):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    # Log transformation
    input_data["total_rooms"] = np.log(
        input_data["total_rooms"] + 1
    )

    input_data["total_bedrooms"] = np.log(
        input_data["total_bedrooms"] + 1
    )

    input_data["population"] = np.log(
        input_data["population"] + 1
    )

    input_data["households"] = np.log(
        input_data["households"] + 1
    )

    # One-hot encoding
    input_data = input_data.join(
        pd.get_dummies(input_data["ocean_proximity"])
    ).drop(
        ["ocean_proximity"],
        axis=1
    )

    # Feature engineering
    input_data["bedroom_ratio"] = (
        input_data["total_bedrooms"] /
        input_data["total_rooms"]
    )

    input_data["household_rooms"] = (
        input_data["total_rooms"] /
        input_data["households"]
    )

    # Match training features
    input_data = input_data.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Result
    st.success(
        f"💰 Estimated Median House Value: ${prediction:,.2f}"
    )

# =========================
# Model information
# =========================

st.divider()

st.caption(
    "🤖 Model: Random Forest Regressor | "
    "🛠️ Python • Scikit-learn • Streamlit"
)