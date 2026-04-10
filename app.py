import streamlit as st
import numpy as np
import joblib
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("lightgbm_final.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.title("🌾 Crop Yield Prediction App")
st.markdown("### Smart Agriculture using AI")
st.write("Predict crop yield and get recommendations for better farming decisions.")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🌱 Enter Farm Conditions")

year = st.sidebar.slider("Year", 2000, 2030, 2024)

fertilizer = st.sidebar.slider("Fertilizer", 0.0, 500.0, 50.0)
pesticide = st.sidebar.slider("Pesticide", 0.0, 500.0, 20.0)

avg_temp_c = st.sidebar.slider("Avg Temperature (°C)", 0.0, 50.0, 25.0)
total_rainfall_mm = st.sidebar.slider("Rainfall (mm)", 0.0, 5000.0, 1000.0)

n = st.sidebar.slider("Nitrogen (N)", 0.0, 200.0, 50.0)
p = st.sidebar.slider("Phosphorus (P)", 0.0, 200.0, 30.0)
k = st.sidebar.slider("Potassium (K)", 0.0, 200.0, 40.0)

ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)

state = st.sidebar.selectbox("State", [
    "arunachal_pradesh", "assam", "bihar", "chhattisgarh", "delhi",
    "gujarat", "haryana", "jharkhand", "karnataka", "maharashtra",
    "meghalaya", "mizoram", "nagaland", "odisha", "tamil_nadu",
    "tripura", "uttarakhand", "west_bengal"
])

season = st.sidebar.selectbox("Season", [
    "kharif", "rabi", "summer", "whole_year", "winter"
])

# -----------------------------
# FEATURE ENGINEERING
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
# ENCODING FEATURES
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
# BUILD INPUT DICTIONARY
# -----------------------------
input_dict = {
    "year": year,
    "fertilizer": fertilizer,
    "pesticide": pesticide,
    "avg_temp_c": avg_temp_c,
    "total_rainfall_mm": total_rainfall_mm,
    "n": n,
    "p": p,
    "k": k,
    "ph": ph,
    "soil_fertility": soil_fertility,
    "rainfall_cat": rainfall_cat,
    "temp_cat": temp_cat,
    "rain_temp_interaction": rain_temp_interaction,
    "fert_ph_interaction": fert_ph_interaction,
    "nitrogen_rain_interaction": nitrogen_rain_interaction,
    "rainfall_log": rainfall_log,
    "fertilizer_log": fertilizer_log
}

# initialize one-hot
for col in state_features + season_features:
    input_dict[col] = 0

input_dict[f"state_{state}"] = 1
input_dict[f"season_{season}"] = 1

# full feature order (VERY IMPORTANT)
feature_order = [
    'year', 'fertilizer', 'pesticide', 'avg_temp_c',
    'total_rainfall_mm', 'n', 'p', 'k', 'ph',
    'soil_fertility', 'rainfall_cat', 'temp_cat',
    'rain_temp_interaction', 'fert_ph_interaction',
    'nitrogen_rain_interaction', 'rainfall_log',
    'fertilizer_log'
] + state_features + season_features

# -----------------------------
# PREDICTION
# -----------------------------
col1, col2 = st.columns(2)

if st.sidebar.button("🚀 Predict Yield"):

    # ✅ FIXED (NO WARNING VERSION)
    input_data = pd.DataFrame([input_dict])[feature_order]

    prediction = model.predict(input_data)[0]

    # -----------------------------
    # RESULT
    # -----------------------------
    with col1:
        st.metric("🌱 Predicted Yield", f"{prediction:.2f}")

        if prediction < 2:
            st.error("Low yield expected ⚠️")
        elif prediction < 5:
            st.warning("Moderate yield ⚖️")
        else:
            st.success("High yield expected ✅")

        st.subheader("💡 Recommendations")

        recs = []

        if n < 40:
            recs.append("Increase Nitrogen")
        if p < 30:
            recs.append("Increase Phosphorus")
        if k < 40:
            recs.append("Increase Potassium")

        if ph < 5.5:
            recs.append("Soil too acidic → add lime")
        elif ph > 7.5:
            recs.append("Soil too alkaline → add organic matter")

        if total_rainfall_mm < 500:
            recs.append("Increase irrigation")
        elif total_rainfall_mm > 3000:
            recs.append("Improve drainage")

        if fertilizer < 20:
            recs.append("Increase fertilizer")
        elif fertilizer > 300:
            recs.append("Reduce fertilizer")

        if pesticide > 200:
            recs.append("Reduce pesticide use")

        if recs:
            for r in recs:
                st.write(f"✅ {r}")
        else:
            st.success("Optimal conditions detected!")

        # download report
        st.subheader("📄 Download Report")

        df = pd.DataFrame({
            "Feature": feature_order,
            "Value": [input_dict[col] for col in feature_order]
        })

        df["Predicted Yield"] = prediction

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "crop_yield_report.csv",
            "text/csv"
        )

    # -----------------------------
    # CHARTS
    # -----------------------------
    with col2:
        st.subheader("📊 Input Overview")

        st.bar_chart({
            "Fertilizer": fertilizer,
            "Rainfall": total_rainfall_mm,
            "Temperature": avg_temp_c,
            "Nitrogen": n,
            "Phosphorus": p,
            "Potassium": k
        })

        st.subheader("📈 Yield Comparison")

        st.bar_chart({
            "Predicted": prediction,
            "Target": 6
        })

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Developed using LightGBM + Streamlit")