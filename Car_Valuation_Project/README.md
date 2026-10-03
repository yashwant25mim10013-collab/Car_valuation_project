#  Car Valuation Prediction

An end-to-end machine learning project that predicts the price (in ₹ lakh) of a used car from its features, with a live web app built using Streamlit.

> **Technical Team Task Round: AI/ML Domain, Task 1** (Exploratory Data Analysis and Predictive Modeling: Tabular Regression)

 **Live demo:** [Open the app](https://YOUR-APP-NAME.streamlit.app)

---

##  Features

- **Data cleaning and preprocessing:** median / most-frequent imputation, IQR outlier capping, one-hot encoding, feature scaling
- **Exploratory Data Analysis:** univariate, bivariate and correlation analysis (histograms, box plots, scatter plots, heatmap)
- **Feature engineering:** `car_age` and `km_per_year`
- **Model training and comparison:** Linear Regression, Random Forest and Gradient Boosting, with 5-fold cross-validation
- **Evaluation metrics:** RMSE, MAE and R²
- **Leakage-free design:** train/test split first, every preprocessing step is fitted on the training set only (scikit-learn `Pipeline`)
- **Log-transformed target:** price is right-skewed, so models are trained on `log(1 + price)` and predictions are converted back to ₹ lakh
- **Interactive web app:** enter a car's details and get an instant price prediction

##  Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-Learn · Streamlit

##  Project Structure

```
Car_valuation_project/
├── app.py                          # Streamlit web app
├── car_model.py                    # Data, preprocessing pipeline and training code used by the app
├── Car_Valuation_Prediction.ipynb  # Main notebook: EDA, modeling, evaluation, explanations
├── car_valuation.py                # Script version of the notebook (saves plots to outputs/)
├── cars.csv                        # Dataset
├── outputs/                        # Saved plots and run log
├── requirements.txt
└── README.md
```

##  How to Run

**1. Install the libraries**
```bash
pip install -r requirements.txt
```

**2. Run the web app**
```bash
streamlit run app.py
```
(If `streamlit` is not recognized, use `python -m streamlit run app.py`.)

**3. Or explore the notebook**
```bash
jupyter notebook Car_Valuation_Prediction.ipynb
```

**4. Or run the script**
```bash
python car_valuation.py
```
Plots are saved to `outputs/`, and at the end the script asks for a car's details in the terminal and prints a predicted price.

##  Web App

The app has three tabs:

| Tab | What it shows |
|---|---|
|  **Predict price** | Enter car details in the sidebar and get the predicted price, plus a price-vs-age curve |
|  **Model comparison** | RMSE, MAE and R² for all models, and an actual-vs-predicted plot |
|  **Data exploration** | Dataset preview and key EDA plots |

##  Results

Test-set results from the notebook:

| Model | RMSE (₹ lakh) | MAE (₹ lakh) | R² |
|---|---|---|---|
| Linear Regression | 0.455 | 0.302 | 0.968 |
| Random Forest | 0.680 | 0.430 | 0.929 |
| Gradient Boosting | 0.481 | 0.286 | 0.964 |

**Key takeaways**
- After the log transform of the target, plain Linear Regression is a very strong baseline, because price depends multiplicatively on age and engine size (which is linear in log space).
- Gradient Boosting has the lowest MAE and the highest cross-validation R², meaning the smallest typical errors.
- Car age, brand, engine size and transmission are the main drivers of price.

##  Methodology

| Step | Technique | Reason |
|---|---|---|
| Missing numeric values | Median imputation | Robust to outliers |
| Missing categorical values | Most-frequent imputation | Natural choice for categories |
| Outliers | IQR capping | Limits extreme values without deleting rows |
| Encoding | One-hot encoding | Brand and fuel type have no natural order |
| Scaling | StandardScaler | Comparable feature scales for the linear model |
| Target | `log1p` transform | Price is right-skewed |
| Validation | Split first, 5-fold CV on train | Avoids data leakage |

##  Deployment

This app is deployed for free on Streamlit Community Cloud directly from this GitHub repository.
To deploy your own copy:
1. Fork this repository (it must be public).
2. Sign in at share.streamlit.io with GitHub and click **Create app**.
3. Select the repository, branch `main` and main file `app.py`, then click **Deploy**.

##  Dataset Note

`cars.csv` is a **synthetic dataset** (3000 rows) generated to mimic real-world data, including missing values, outliers and categorical columns. Predicted prices are therefore illustrative, not real market prices.

To use a real dataset (for example the Kaggle *Used Cars* dataset), replace `cars.csv` and update `TARGET`, `num_features` and `cat_features` to match its columns, then re-run the notebook or app.

##  Future Improvements

- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Try XGBoost or LightGBM
- Retrain on a real-world dataset
- Add model explainability (SHAP) to the app

##  Author

**Yashwant Yadav**
GitHub: [yashwant25mim10013-collab](https://github.com/yashwant25mim10013-collab) · LinkedIn: [yashwant-yadav-533362383](https://linkedin.com/in/yashwant-yadav-533362383/)