import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("lightgbm_final.pkl")

st.title("🌾 Crop Yield Prediction App")

st.header("Basic Inputs")

year = st.number_input("Year", 2000, 2030, 2024)
fertilizer = st.number_input("Fertilizer", 0.0)
pesticide = st.number_input("Pesticide", 0.0)
avg_temp_c = st.number_input("Avg Temperature (°C)", 0.0)
total_rainfall_mm = st.number_input("Rainfall (mm)", 0.0)
n = st.number_input("Nitrogen (N)", 0.0)
p = st.number_input("Phosphorus (P)", 0.0)
k = st.number_input("Potassium (K)", 0.0)
ph = st.number_input("Soil pH", 0.0)

state = st.selectbox("State", [
    "arunachal_pradesh", "assam", "bihar", "chhattisgarh", "delhi",
    "gujarat", "haryana", "jharkhand", "karnataka", "maharashtra",
    "meghalaya", "mizoram", "nagaland", "odisha", "tamil_nadu",
    "tripura", "uttarakhand", "west_bengal"
])

season = st.selectbox("Season", [
    "kharif", "rabi", "summer", "whole_year", "winter"
])

# -----------------------------
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# -----------------------------
soil_fertility = (n + p + k) / 3

rainfall_cat = 0 if total_rainfall_mm < 500 else 1 if total_rainfall_mm < 1500 else 2
temp_cat = 0 if avg_temp_c < 20 else 1 if avg_temp_c <= 30 else 2

rain_temp_interaction = total_rainfall_mm * avg_temp_c
fert_ph_interaction = fertilizer * ph
nitrogen_rain_interaction = n * total_rainfall_mm

rainfall_log = np.log1p(total_rainfall_mm)
fertilizer_log = np.log1p(fertilizer)

# -----------------------------
# STATE ENCODING (39 FEATURES IN YOUR LIST)
# -----------------------------
state_features = [
    'state_arunachal_pradesh', 'state_assam', 'state_bihar',
    'state_chhattisgarh', 'state_delhi', 'state_gujarat',
    'state_haryana', 'state_jharkhand', 'state_karnataka',
    'state_maharashtra', 'state_meghalaya', 'state_mizoram',
    'state_nagaland', 'state_odisha', 'state_tamil_nadu',
    'state_tripura', 'state_uttarakhand', 'state_west_bengal'
]

season_features = [
    'season_kharif', 'season_rabi', 'season_summer',
    'season_whole_year', 'season_winter'
]

# -----------------------------
# BUILD INPUT VECTOR (40+ FEATURES)
# -----------------------------
input_dict = {}

# base features
input_dict["year"] = year
input_dict["fertilizer"] = fertilizer
input_dict["pesticide"] = pesticide
input_dict["avg_temp_c"] = avg_temp_c
input_dict["total_rainfall_mm"] = total_rainfall_mm
input_dict["n"] = n
input_dict["p"] = p
input_dict["k"] = k
input_dict["ph"] = ph

# engineered features
input_dict["soil_fertility"] = soil_fertility
input_dict["rainfall_cat"] = rainfall_cat
input_dict["temp_cat"] = temp_cat
input_dict["rain_temp_interaction"] = rain_temp_interaction
input_dict["fert_ph_interaction"] = fert_ph_interaction
input_dict["nitrogen_rain_interaction"] = nitrogen_rain_interaction
input_dict["rainfall_log"] = rainfall_log
input_dict["fertilizer_log"] = fertilizer_log

# initialize all state + season columns as 0
for col in state_features + season_features:
    input_dict[col] = 0

# activate selected state
state_col = f"state_{state}"
if state_col in input_dict:
    input_dict[state_col] = 1

# activate selected season
season_col = f"season_{season}"
if season_col in input_dict:
    input_dict[season_col] = 1

# -----------------------------
# FINAL INPUT ARRAY (MATCH MODEL ORDER)
# -----------------------------
feature_order = [
    'year', 'fertilizer', 'pesticide', 'avg_temp_c',
    'total_rainfall_mm', 'n', 'p', 'k', 'ph',
    'soil_fertility', 'rainfall_cat', 'temp_cat',
    'rain_temp_interaction', 'fert_ph_interaction',
    'nitrogen_rain_interaction', 'rainfall_log',
    'fertilizer_log'
] + state_features + season_features

if st.button("Predict Yield"):
    input_data = np.array([[input_dict[col] for col in feature_order]])

    prediction = model.predict(input_data)

    st.success(f"🌱 Predicted Yield: {prediction[0]:.2f}")