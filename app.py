import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Charger le modèle et le scaler
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Titre
st.title("🏠 House Price Predictor")
st.write("Predict the median house value using a Machine Learning model.")

# Informations de la maison / zone
st.subheader("Enter house information")

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)

housing_median_age = st.number_input(
    "Housing Median Age",
    value=41.0
)

total_rooms = st.number_input(
    "Total Rooms in Area",
    value=880.0
)

total_bedrooms = st.number_input(
    "Total Bedrooms in Area",
    value=129.0
)

population = st.number_input(
    "Population in Area",
    value=322.0
)

households = st.number_input(
    "Households in Area",
    value=126.0
)

median_income = st.number_input(
    "Median Income",
    value=8.3
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)
if st.button("Predict Price"):

    # Créer les données d'entrée
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

    # Transformations logarithmiques
    input_data["total_rooms"] = np.log(input_data["total_rooms"] + 1)
    input_data["total_bedrooms"] = np.log(input_data["total_bedrooms"] + 1)
    input_data["population"] = np.log(input_data["population"] + 1)
    input_data["households"] = np.log(input_data["households"] + 1)

    # One-hot encoding
    input_data = input_data.join(
        pd.get_dummies(input_data["ocean_proximity"])
    ).drop(["ocean_proximity"], axis=1)

    # Ajouter les colonnes manquantes
    input_data = input_data.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    # Feature engineering
    input_data["bedroom_ratio"] = (
        input_data["total_bedrooms"] / input_data["total_rooms"]
    )

    input_data["household_rooms"] = (
        input_data["total_rooms"] / input_data["households"]
    )

    # Remettre les colonnes dans le même ordre que pendant l'entraînement
    input_data = input_data.reindex(
        columns=scaler.feature_names_in_,
        fill_value=0
    )

    # Standardisation
    input_scaled = scaler.transform(input_data)

    # Prédiction
    prediction = model.predict(input_scaled)[0]

    # Afficher le résultat
    st.success(f"💰 Estimated Median House Value: ${prediction:,.2f}")