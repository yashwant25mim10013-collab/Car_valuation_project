# Car Valuation Prediction
**Technical Team Task Round - AI/ML Domain, Task 1: Exploratory Data Analysis and Predictive Modeling (Tabular Regression)**

An end-to-end machine learning pipeline that predicts the price (in lakh INR) of a used car from its features.

## How to run
```
pip install -r requirements.txt
jupyter notebook Car_Valuation_Prediction.ipynb     # notebook with plots and explanations
python car_valuation.py                             # script version (plots are saved to outputs/)
```

## What is included
- **Data cleaning and preprocessing:** median / most-frequent imputation, IQR outlier capping, One-Hot encoding, StandardScaler
- **EDA:** univariate, bivariate and correlation analysis (histograms, box plots, scatter plots, heatmap)
- **Models:** Linear Regression, Random Forest, Gradient Boosting (with 5-fold cross-validation)
- **Metrics:** RMSE, MAE, R-squared
- **Leakage-free design:** train/test split first, all preprocessing fitted on the training set only (scikit-learn Pipeline)
- **Prediction for a new car:** `predict_car(...)` in the last notebook cell, or the interactive prompt at the end of the script

## Predicting the price of a new car
- Script: after training, `python car_valuation.py` asks for the car details in the terminal (press Enter to use the default).
- Notebook: change the values in the last cell and run it.
- Supported brands: Audi, BMW, Honda, Hyundai, Mahindra, Maruti, Mercedes, Tata, Toyota.

## Web app (Streamlit)
```
streamlit run app.py
```
The app has three tabs: **Predict price** (enter car details in the sidebar), **Model comparison** (RMSE, MAE, R² and an actual-vs-predicted plot) and **Data exploration** (EDA plots).
It uses `car_model.py` (data, preprocessing pipeline and training code) and trains the models on first load.

### Free deployment on Streamlit Community Cloud
1. Push this folder to a public GitHub repository (`app.py`, `car_model.py`, `cars.csv` and `requirements.txt` must be in the repo root).
2. Sign in at share.streamlit.io with your GitHub account and click **Create app**.
3. Select the repository, the `main` branch and `app.py` as the main file, then click **Deploy**.
4. After a few minutes you get a public link you can share.

## Dataset
`cars.csv` contains a synthetic dataset (3000 rows with missing values and outliers). To use a real dataset
(for example the Kaggle *Used Cars* dataset), replace `cars.csv` and update `TARGET`, `num_features` and `cat_features` to match its columns.
