import os

import numpy as np
import pandas as pd
import streamlit as st
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "airbnb_price_pipeline.pkl")

st.set_page_config(page_title="Airbnb NYC Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 Airbnb NYC — Nightly Price Predictor")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file `airbnb_price_pipeline.pkl` was not found in this folder.\n\n"
        "Run `Airbnb_Price_Prediction.ipynb` end-to-end first (with `AB_NYC_2019.csv` "
        "in this same folder) — the last cell saves the trained pipeline here."
    )
    st.stop()


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

st.write("Fill in the listing details below to get an estimated nightly price.")

col1, col2 = st.columns(2)
with col1:
    neighbourhood_group = st.selectbox(
        "Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )
    room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])
    minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=3)
    availability_365 = st.slider("Days available per year", 0, 365, 180)

with col2:
    latitude = st.number_input("Latitude", min_value=40.4, max_value=41.0, value=40.73, format="%.5f")
    longitude = st.number_input("Longitude", min_value=-74.3, max_value=-73.6, value=-73.99, format="%.5f")
    number_of_reviews = st.number_input("Number of reviews", min_value=0, max_value=2000, value=10)
    reviews_per_month = st.number_input("Reviews per month", min_value=0.0, max_value=30.0, value=1.0, step=0.1)

calculated_host_listings_count = st.slider("Host's total number of listings", 1, 100, 1)
days_since_last_review = st.slider(
    "Days since last review (use a large number like 365+ if no reviews yet)",
    0, 2000, 60,
)

st.divider()

if st.button("Predict nightly price", type="primary"):
    input_df = pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "days_since_last_review": days_since_last_review,
        "neighbourhood_group": neighbourhood_group,
        "room_type": room_type,
    }])

    log_pred = pipeline.predict(input_df)[0]
    price_pred = float(np.expm1(log_pred))

    st.metric("Estimated nightly price", f"${price_pred:,.0f}")
    st.caption(
        "This is a point estimate from a model trained on 2019 NYC listings — treat it as "
        "a ballpark reference, not an exact valuation."
    )

    with st.expander("Show the exact input sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
with st.expander("About this app"):
    st.write(
        "This app loads a scikit-learn pipeline (preprocessing + regression model) trained "
        "in `Airbnb_Price_Prediction.ipynb` on the Kaggle 'New York City Airbnb Open Data' "
        "dataset (`AB_NYC_2019.csv`). The pipeline handles feature scaling and one-hot "
        "encoding internally, so this app only needs to pass in raw values."
    )
