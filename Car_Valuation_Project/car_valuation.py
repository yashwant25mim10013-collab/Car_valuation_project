import matplotlib
matplotlib.use("Agg")      # save plots to files instead of opening windows
import os
os.makedirs("outputs", exist_ok=True)   # create the outputs folder automatically
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid", palette="deep")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "cars.csv"     # <-- put the path to your CSV here (optional)
TARGET = "price_lakh"

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
    price = (base
             * np.exp(-0.085 * age)
             * (engine / 1200) ** 0.55
             * np.where(fuel == "Diesel", 1.12, np.where(fuel == "Electric", 1.45,
                        np.where(fuel == "CNG", 0.95, 1.0)))
             * np.where(trans == "Automatic", 1.18, 1.0)
             * np.where(owner == "First", 1.0, np.where(owner == "Second", 0.88, 0.76))
             * np.exp(-km / 650000)
             * np.exp(rng.normal(0, 0.10, n)))

    df = pd.DataFrame({"brand": brand, "model_year": year, "fuel_type": fuel,
                       "transmission": trans, "owner_type": owner, "km_driven": km.round(0),
                       "engine_cc": engine, "power_bhp": power.round(1),
                       "mileage_kmpl": mileage.round(1), "seats": seats,
                       TARGET: price.round(2)})

    # Real-world mess: outliers (data-entry errors) and missing values
    out_idx = rng.choice(n, int(0.012 * n), replace=False)
    df.loc[out_idx, "km_driven"] *= rng.integers(8, 20, len(out_idx))
    for col, frac in [("mileage_kmpl", .05), ("power_bhp", .04), ("engine_cc", .03),
                      ("seats", .02), ("fuel_type", .015)]:
        df.loc[rng.choice(n, int(frac * n), replace=False), col] = np.nan
    return df

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from: {DATA_PATH}")
else:
    df = generate_synthetic_cars()
    print("CSV not found -> generated a synthetic dataset.")

print("Shape:", df.shape)
df.head()

df.info()
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])
df.describe().T

# Target distribution: it is skewed, which justifies a log transform later
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[TARGET], kde=True, ax=ax[0]).set_title("Price (lakh ₹) - right-skewed")
sns.histplot(np.log1p(df[TARGET]), kde=True, ax=ax[1], color="green").set_title("log(1 + Price) - much more symmetric")
plt.tight_layout(); plt.savefig("outputs/plot_01.png", dpi=120, bbox_inches="tight"); plt.close()
print("Skewness  raw:", round(df[TARGET].skew(), 2), "| log:", round(np.log1p(df[TARGET]).skew(), 2))

num_cols = ["model_year", "km_driven", "engine_cc", "power_bhp", "mileage_kmpl", "seats"]
cat_cols = ["brand", "fuel_type", "transmission", "owner_type"]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for a, c in zip(axes.ravel(), num_cols):
    sns.histplot(df[c].dropna(), kde=True, ax=a); a.set_title(f"Distribution: {c}")
plt.tight_layout(); plt.savefig("outputs/plot_02.png", dpi=120, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for a, c in zip(axes, cat_cols):
    order = df[c].value_counts().index
    sns.countplot(data=df, x=c, order=order, ax=a); a.set_title(f"Count: {c}")
    a.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("outputs/plot_03.png", dpi=120, bbox_inches="tight"); plt.close()

# Outlier detection with box plots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for a, c in zip(axes, ["km_driven", "engine_cc", "power_bhp", "mileage_kmpl"]):
    sns.boxplot(y=df[c], ax=a, color="skyblue"); a.set_title(f"Box plot: {c}")
plt.tight_layout(); plt.savefig("outputs/plot_04.png", dpi=120, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
sns.scatterplot(data=df.assign(age=2026 - df.model_year), x="age", y=TARGET, alpha=.3, ax=axes[0]).set_title("Price vs Car Age")
sns.scatterplot(data=df, x="km_driven", y=TARGET, alpha=.3, ax=axes[1]).set_title("Price vs KM Driven")
sns.scatterplot(data=df, x="engine_cc", y=TARGET, alpha=.3, ax=axes[2]).set_title("Price vs Engine CC")
plt.tight_layout(); plt.savefig("outputs/plot_05.png", dpi=120, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for a, c in zip(axes, cat_cols):
    order = df.groupby(c)[TARGET].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=c, y=TARGET, order=order, ax=a); a.set_title(f"Price by {c}")
    a.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("outputs/plot_06.png", dpi=120, bbox_inches="tight"); plt.close()

plt.figure(figsize=(8, 6))
corr = df[num_cols + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap (numeric features)")
plt.savefig("outputs/plot_07.png", dpi=120, bbox_inches="tight"); plt.close()

def add_features(d):
    d = d.copy()
    d["car_age"] = 2026 - d["model_year"]
    d["km_per_year"] = d["km_driven"] / d["car_age"].clip(lower=1)
    return d.drop(columns=["model_year"])

data = add_features(df)
X = data.drop(columns=[TARGET])
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "| Test:", X_test.shape)

num_features = ["km_driven", "engine_cc", "power_bhp", "mileage_kmpl", "seats", "car_age", "km_per_year"]
cat_features = ["brand", "fuel_type", "transmission", "owner_type"]

class IQRClipper(BaseEstimator, TransformerMixin):
    """Learns outlier bounds from the training data only, then clips both train and test."""
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

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clip", IQRClipper(factor=1.5)),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, num_features),
    ("cat", categorical_pipe, cat_features),
])
preprocessor

def make_model(estimator):
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor), ("model", estimator)]),
        func=np.log1p, inverse_func=np.expm1,
    )

models = {
    "Linear Regression": make_model(LinearRegression()),
    "Random Forest": make_model(RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                       n_jobs=-1, random_state=RANDOM_STATE)),
    "Gradient Boosting": make_model(GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                                               max_depth=3, random_state=RANDOM_STATE)),
}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, preds = [], {}

for name, model in models.items():
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    results.append({
        "Model": name,
        "CV R² (train)": round(cv_r2, 4),
        "Test RMSE": round(np.sqrt(mean_squared_error(y_test, p)), 4),
        "Test MAE": round(mean_absolute_error(y_test, p), 4),
        "Test R²": round(r2_score(y_test, p), 4),
    })

results_df = pd.DataFrame(results).set_index("Model")
results_df

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for a, (metric, color) in zip(axes, [("Test RMSE", "tomato"), ("Test MAE", "orange"), ("Test R²", "seagreen")]):
    results_df[metric].plot(kind="bar", ax=a, color=color)
    a.set_title(metric); a.set_xlabel(""); a.tick_params(axis="x", rotation=20)
    for i, v in enumerate(results_df[metric]):
        a.text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig("outputs/plot_08.png", dpi=120, bbox_inches="tight"); plt.close()

best_name = results_df["Test R²"].idxmax()
print(f"Best model (highest test R²): {best_name}")

best_pred = preds[best_name]
residuals = y_test - best_pred

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].scatter(y_test, best_pred, alpha=.4)
lims = [0, max(y_test.max(), best_pred.max())]
axes[0].plot(lims, lims, "r--"); axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Actual vs Predicted - {best_name}")

axes[1].scatter(best_pred, residuals, alpha=.4); axes[1].axhline(0, color="r", ls="--")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual"); axes[1].set_title("Residual Plot")

sns.histplot(residuals, kde=True, ax=axes[2]); axes[2].set_title("Residual Distribution")
plt.tight_layout(); plt.savefig("outputs/plot_09.png", dpi=120, bbox_inches="tight"); plt.close()

best_model = models[best_name]
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                              random_state=RANDOM_STATE, scoring="r2")
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()

imp.plot(kind="barh", figsize=(8, 5), color="steelblue")
plt.title(f"Permutation Importance - {best_name}"); plt.xlabel("Drop in R²")
plt.savefig("outputs/plot_10.png", dpi=120, bbox_inches="tight"); plt.close()

def predict_car(brand, model_year, fuel_type, transmission, owner_type,
                km_driven, engine_cc, power_bhp, mileage_kmpl, seats, model=None):
    """Takes the details of one car and returns the predicted price in lakh Rs."""
    model = model or best_model
    row = pd.DataFrame([{
        "brand": brand, "model_year": model_year, "fuel_type": fuel_type,
        "transmission": transmission, "owner_type": owner_type,
        "km_driven": km_driven, "engine_cc": engine_cc, "power_bhp": power_bhp,
        "mileage_kmpl": mileage_kmpl, "seats": seats,
    }])
    row = add_features(row)[X.columns]      # same feature engineering as in training
    return float(model.predict(row)[0])     # the log-transform inverse is applied automatically

# ---- Change the values below to your own car ----
price = predict_car(brand="Toyota", model_year=2020, fuel_type="Diesel",
                    transmission="Automatic", owner_type="First", km_driven=30000,
                    engine_cc=2000, power_bhp=148, mileage_kmpl=14, seats=7)
print(f"Predicted price: Rs {price:.2f} lakh")


# ---- Interactive: enter your own car details in the terminal (press Enter for the default) ----
def ask(prompt, default, cast=str, options=None):
    hint = f" {options}" if options else ""
    val = input(f"{prompt}{hint} [{default}]: ").strip()
    return cast(val) if val else default

if __name__ == "__main__":
    print("\n=== Enter your car details (press Enter to accept the default) ===")
    my_price = predict_car(
        brand=ask("Brand", "Maruti", str, sorted(df['brand'].dropna().unique())),
        model_year=ask("Model year", 2018, int),
        fuel_type=ask("Fuel type", "Petrol", str, ["Petrol", "Diesel", "CNG", "Electric"]),
        transmission=ask("Transmission", "Manual", str, ["Manual", "Automatic"]),
        owner_type=ask("Owner type", "First", str, ["First", "Second", "Third+"]),
        km_driven=ask("KM driven", 50000, float),
        engine_cc=ask("Engine (cc)", 1200, float),
        power_bhp=ask("Power (bhp)", 85, float),
        mileage_kmpl=ask("Mileage (kmpl)", 18, float),
        seats=ask("Seats", 5, int),
    )
    print(f"\n>>> Predicted price: Rs {my_price:.2f} lakh")
