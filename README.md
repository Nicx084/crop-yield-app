# crop-yield-app
Crop yield prediction web app using LightGBM and Streamlit
# 🌾 Crop Yield Prediction System

## Project Overview

This project is an AI-powered system designed to predict crop yield using environmental and soil data. It integrates machine learning, data transformation, and interactive visualization to support smarter agricultural decision-making.

The system analyzes factors such as rainfall, temperature, soil nutrients, and seasonal patterns to estimate crop productivity and uncover meaningful insights.
## 🎯 Objectives
* Predict crop yield using machine learning (LightGBM model)
* Analyze the impact of environmental and soil factors
* Build an interactive application for real-time predictions
* Visualize agricultural data for insights and decision-making
## Features
* 🌦 Yield prediction based on user inputs
* 🌱 Soil fertility analysis using N, P, K
* 📊 Interactive dashboard (Power BI)
* 📈 Key visualizations:
  * Yield vs Rainfall
  * Yield vs Temperature
  * Yield vs Soil Fertility
  * Seasonal trends
  * State-wise comparison
  * Yield distribution
##  Machine Learning Model
* Model Used: LightGBM Regressor
* Task: Regression (predicting continuous crop yield values)
### Input Features:
* Year
* Rainfall
* Temperature
* Fertilizer usage
* Pesticide usage
* Soil nutrients (N, P, K)
* Soil pH
* Season
* State
The model learns patterns from historical data to make accurate predictions of crop yield under different conditions.
## 🔄 Data Transformation & Feature Engineering
### Original Dataset
* Contained 43 columns with one-hot encoded flags for states and seasons (e.g., `state_tamil nadu`, `season_kharif`)
* Each row had multiple `True/False` indicators, making the dataset wide and harder to interpret
### Transformed Dataset
* Converted into a **tidy format** with categorical columns:
  * `state` (e.g., Assam, Karnataka, West Bengal)
  * `season` (e.g., Kharif, Rabi, Summer, Whole Year)
* Added **engineered features**:
  * `soil_fertility` (derived from NPK values)
  * `rainfall_cat`, `temp_cat` (categorical bins)
  * Interaction terms:
    * `rain_temp_interaction`
    * `fert_ph_interaction`
    * `nitrogen_rain_interaction`
  * Log transformations:
    * `rainfall_log`
    * `fertilizer_log`
### Importance
* **Cleaner for Dashboards**: Easier slicing by state and season in Power BI
* **Better for ML Models**: Captures non-linear relationships
* **Improved Interpretability**: Easier to explain to stakeholders
* **Portfolio Value**: Demonstrates practical feature engineering skills
##  Application
An interactive application was developed using Streamlit, allowing users to:
* Input farm conditions (rainfall, temperature, soil data, etc.)
* Get real-time crop yield predictions
* Understand how different factors influence productivity
##  Data Visualization
Power BI was used to create dashboards that highlight:
* Environmental impact on yield
* Seasonal trends
* Regional (state-wise) performance
* Yield distribution patterns
These visualizations support better understanding and decision-making.
##  Usefulness
* Helps farmers optimize inputs for higher productivity
* Supports climate-smart agriculture
* Assists policymakers in planning food supply
* Encourages data-driven farming practices
##  Future Improvements
* Use local (Kenya) agricultural data
* Integrate real-time weather data
* Improve model accuracy
* Add fertilizer recommendation system
##  Conclusion
This project demonstrates how machine learning, feature engineering, and data visualization can be applied in agriculture to improve productivity and decision-making. It highlights the role of AI in transforming traditional farming into a smarter and more efficient system.
## 👥 Contributors
1. Jacqyline Mwaura
2. Nicholas Mutuku  
3. Brian Kimutai
