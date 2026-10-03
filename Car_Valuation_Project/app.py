"""Streamlit web app: Car Valuation Prediction. Run with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import car_model as cm

st.set_page_config(page_title="Car Valuation Predictor", page_icon=" ", layout="wide")
sns.set_theme(style="whitegrid", palette="deep")


@st.cache_resource(show_spinner="Training models (first load only)...")
def get_models():
    df = cm.load_data()
    models, results, split = cm.train_all(df)
    return df, models, results, split


df, models, results, (X_train, X_test, y_train, y_test) = get_models()
columns = list(X_train.columns)

st.title(" Car Valuation Predictor")
st.caption("Predict the price of a used car (in ₹ lakh) with machine learning. "
           "Note: the model is trained on a synthetic dataset, so prices are illustrative.")

# ------------------------------ Sidebar: car details ------------------------------
st.sidebar.header("Car details")
model_name = st.sidebar.selectbox("Model", list(models), index=list(models).index("Gradient Boosting"))
brand = st.sidebar.selectbox("Brand", sorted(df["brand"].dropna().unique()), index=5)
model_year = st.sidebar.slider("Model year", 2005, 2026, 2019)
fuel_type = st.sidebar.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "Electric"])
transmission = st.sidebar.radio("Transmission", ["Manual", "Automatic"], horizontal=True)
owner_type = st.sidebar.selectbox("Owner type", ["First", "Second", "Third+"])
km_driven = st.sidebar.number_input("Kilometres driven", 500, 500000, 45000, step=1000)
engine_cc = st.sidebar.select_slider("Engine (cc)", [800, 1000, 1200, 1500, 1800, 2000, 2500, 3000], value=1200)
power_bhp = st.sidebar.slider("Power (bhp)", 40, 300, 85)
mileage_kmpl = st.sidebar.slider("Mileage (kmpl)", 5.0, 30.0, 18.0, step=0.5)
seats = st.sidebar.selectbox("Seats", [4, 5, 7], index=1)

tab_pred, tab_models, tab_eda = st.tabs([" Predict price", " Model comparison", " Data exploration"])

# ------------------------------ Tab 1: prediction ------------------------------
with tab_pred:
    price = cm.predict_price(models[model_name], columns, brand, model_year, fuel_type, transmission,
                             owner_type, km_driven, engine_cc, power_bhp, mileage_kmpl, seats)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted price", f"₹ {price:.2f} lakh")
    c2.metric("Car age", f"{2026 - model_year} years")
    c3.metric("Model used", model_name)

    st.subheader("How the price changes with age")
    ages = range(2005, 2027)
    curve = [cm.predict_price(models[model_name], columns, brand, y, fuel_type, transmission, owner_type,
                              km_driven, engine_cc, power_bhp, mileage_kmpl, seats) for y in ages]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(list(ages), curve, marker="o")
    ax.axvline(model_year, color="red", ls="--", label="Selected year")
    ax.set_xlabel("Model year"); ax.set_ylabel("Predicted price (₹ lakh)"); ax.legend()
    st.pyplot(fig)
    st.caption("All other details are kept the same as in the sidebar.")

# ------------------------------ Tab 2: model comparison ------------------------------
with tab_models:
    st.subheader("Test-set performance")
    st.dataframe(results, width="stretch")
    best = results["R²"].idxmax()
    st.success(f"Highest test R²: {best}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (metric, color) in zip(axes, [("RMSE", "tomato"), ("MAE", "orange"), ("R²", "seagreen")]):
        results[metric].plot(kind="bar", ax=ax, color=color)
        ax.set_title(metric); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); st.pyplot(fig)

    st.subheader(f"Actual vs predicted ({model_name})")
    pred = models[model_name].predict(X_test)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(y_test, pred, alpha=.4)
    lim = [0, max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "r--"); ax.set_xlabel("Actual (₹ lakh)"); ax.set_ylabel("Predicted (₹ lakh)")
    st.pyplot(fig)
    st.markdown("**Method:** train/test split first, then imputation, IQR outlier capping, one-hot encoding and "
                "scaling are all fitted on the training set only (no data leakage). The target is log-transformed "
                "because price is right-skewed.")

# ------------------------------ Tab 3: EDA ------------------------------
with tab_eda:
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), width="stretch")
    st.caption(f"{df.shape[0]} rows, {df.shape[1]} columns. Missing values per column: "
               + ", ".join(f"{k}: {v}" for k, v in df.isna().sum().items() if v > 0))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.histplot(df[cm.TARGET], kde=True, ax=ax); ax.set_title("Price distribution (right-skewed)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.scatterplot(data=df.assign(age=2026 - df.model_year), x="age", y=cm.TARGET, alpha=.3, ax=ax)
        ax.set_title("Price vs car age"); st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        order = df.groupby("brand")[cm.TARGET].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="brand", y=cm.TARGET, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=45); ax.set_title("Price by brand"); st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        num = ["model_year", "km_driven", "engine_cc", "power_bhp", "mileage_kmpl", "seats", cm.TARGET]
        sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
        ax.set_title("Correlation heatmap"); st.pyplot(fig)
