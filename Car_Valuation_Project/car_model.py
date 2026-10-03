"""Reusable model code for the Streamlit app: data loading, preprocessing pipeline, training."""
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
TARGET = "price_lakh"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cars.csv")

NUM_FEATURES = ["km_driven", "engine_cc", "power_bhp", "mileage_kmpl", "seats", "car_age", "km_per_year"]
CAT_FEATURES = ["brand", "fuel_type", "transmission", "owner_type"]


def generate_synthetic_cars(n=3000, seed=RANDOM_STATE):
    """Realistic used-car dataset with missing values, outliers and categorical columns."""
    rng = np.random.default_rng(seed)
    brands = {"Maruti": 0.9, "Hyundai": 1.0, "Honda": 1.15, "Toyota": 1.35,
              "Mahindra": 1.25, "Tata": 1.0, "BMW": 2.8, "Audi": 2.7, "Mercedes": 3.0}
    brand = rng.choice(list(brands), n, p=[.24, .20, .10, .10, .10, .10, .06, .05, .05])
    year = rng.integers(2005, 2026, n)
    age = 2026 - year
    fuel = rng.choice(["Petrol", "Diesel", "CNG", "Electric"], n, p=[.50, .38, .08, .04])
    trans = rng.choice(["Manual", "Automatic"], n, p=[.65, .35])
    owner = rng.choice(["First", "Second", "Third+"], n, p=[.65, .27, .08])
    engine = rng.choice([800, 1000, 1200, 1500, 1800, 2000, 2500, 3000], n,
                        p=[.08, .15, .25, .20, .12, .12, .05, .03])
    power = engine / 13.5 + rng.normal(0, 6, n)
    mileage = 28 - engine / 160 + rng.normal(0, 1.8, n)
    seats = rng.choice([4, 5, 7], n, p=[.05, .75, .20])
    km = np.clip(rng.normal(11000, 4500, n) * np.maximum(age, 0.5) + rng.normal(0, 5000, n), 500, None)
    base = 5.5 * np.array([brands[b] for b in brand])
    price = (base * np.exp(-0.085 * age) * (engine / 1200) ** 0.55
             * np.where(fuel == "Diesel", 1.12, np.where(fuel == "Electric", 1.45,
                        np.where(fuel == "CNG", 0.95, 1.0)))
             * np.where(trans == "Automatic", 1.18, 1.0)
             * np.where(owner == "First", 1.0, np.where(owner == "Second", 0.88, 0.76))
             * np.exp(-km / 650000) * np.exp(rng.normal(0, 0.10, n)))
    df = pd.DataFrame({"brand": brand, "model_year": year, "fuel_type": fuel,
                       "transmission": trans, "owner_type": owner, "km_driven": km.round(0),
                       "engine_cc": engine, "power_bhp": power.round(1),
                       "mileage_kmpl": mileage.round(1), "seats": seats, TARGET: price.round(2)})
    out_idx = rng.choice(n, int(0.012 * n), replace=False)
    df.loc[out_idx, "km_driven"] *= rng.integers(8, 20, len(out_idx))
    for col, frac in [("mileage_kmpl", .05), ("power_bhp", .04), ("engine_cc", .03),
                      ("seats", .02), ("fuel_type", .015)]:
        df.loc[rng.choice(n, int(frac * n), replace=False), col] = np.nan
    return df


def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else generate_synthetic_cars()


def add_features(d):
    d = d.copy()
    d["car_age"] = 2026 - d["model_year"]
    d["km_per_year"] = d["km_driven"] / d["car_age"].clip(lower=1)
    return d.drop(columns=["model_year"])


class IQRClipper(BaseEstimator, TransformerMixin):
    """Learns outlier bounds from the training data only, then clips train and test."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.factor * iqr, q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor():
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("clip", IQRClipper(1.5)), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric_pipe, NUM_FEATURES), ("cat", categorical_pipe, CAT_FEATURES)])


def make_model(estimator):
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", build_preprocessor()), ("model", estimator)]),
        func=np.log1p, inverse_func=np.expm1)


def train_all(df):
    """Split first, then fit every model on the training set only. Returns models, results table, test data."""
    data = add_features(df)
    X, y = data.drop(columns=[TARGET]), data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    models = {
        "Linear Regression": make_model(LinearRegression()),
        "Random Forest": make_model(RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                                           n_jobs=-1, random_state=RANDOM_STATE)),
        "Gradient Boosting": make_model(GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                                                   max_depth=3, random_state=RANDOM_STATE)),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        rows.append({"Model": name,
                     "RMSE": round(float(np.sqrt(mean_squared_error(y_test, p))), 4),
                     "MAE": round(float(mean_absolute_error(y_test, p)), 4),
                     "R²": round(float(r2_score(y_test, p)), 4)})
    return models, pd.DataFrame(rows).set_index("Model"), (X_train, X_test, y_train, y_test)


def predict_price(model, columns, brand, model_year, fuel_type, transmission, owner_type,
                  km_driven, engine_cc, power_bhp, mileage_kmpl, seats):
    row = pd.DataFrame([{"brand": brand, "model_year": model_year, "fuel_type": fuel_type,
                         "transmission": transmission, "owner_type": owner_type, "km_driven": km_driven,
                         "engine_cc": engine_cc, "power_bhp": power_bhp,
                         "mileage_kmpl": mileage_kmpl, "seats": seats}])
    return float(model.predict(add_features(row)[columns])[0])
