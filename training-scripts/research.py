import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from helpers.equipment_beta import *

data = pd.read_csv("data/watch_price_20250910.csv")

df = data.copy()

df["Price (USD)"].mean()
## GEMINI ###

print("--- 1. SÜTUN İSİMLERİ ---")
try:
    features = df.drop('Price', axis=1).columns.tolist()
    print(features)
except KeyError:
    print("Not: 'Price' adında bir sütun bulunamadı. Tüm sütunlar listeleniyor:")
    features = df.columns.tolist()
    print(features)

print("\n--- 2. KATEGORİK SÜTUNLARIN BENZERSİZ DEĞERLERİ ---")
categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

for col in categorical_features:
    if col == 'Price':
        continue

    print(f"\n----- Sütun: {col} -----")
    try:
        unique_values = sorted(df[col].dropna().unique().tolist())
        print(unique_values)
    except TypeError:
        print(f"Sıralanamadı, orjinal sıra: {df[col].dropna().unique().tolist()}")

print("\n--- KOD SONU ---")

# EDA ##
check_data(df, target="Price (USD)")
sns.histplot(data=df, x="Price (USD)")
sns.scatterplot(data=df, x="Price (USD)", y = "Weight (g)")
# --- Outlier Part ---
df.select_dtypes(include=["float64","int64"])

## IQR & ZSCORE for Case Diameter
## --- IQR ---
q1 = df["Case Diameter"].quantile(0.25)
q3 = df["Case Diameter"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df[(df["Case Diameter"] < lower_bound) | (df["Case Diameter"] > upper_bound)]

df[(df["Gender"] == "Male") & (df["Case Diameter"] > 50)]

df[(df["Gender"] == "Unisex") & ((df["Case Diameter"] < 36) | (df["Case Diameter"] > 44))]

## --- ZSCORE ---
series = df["Case Diameter"].dropna()
threshold = 3.0

mean = series.mean()
std = series.std()

z_scores = (series - mean) / std

lower_bound_z = mean - threshold * std
upper_bound_z = mean + threshold * std

series[np.abs(z_scores) > threshold]

sns.histplot(data=df, x="Case Diameter", kde=True)

df["Case Diameter"].skew()

df[df["Case Diameter"] > 50]["Case Diameter"].value_counts().sort_index()

df[(df["Gender"] == "Female") & (df["Case Diameter"] > 40)]


# --- Solutions ---
df = df[~((df["Gender"] == "Female") & (df["Case Diameter"] > 40))]
df = df[~((df["Gender"] == "Unisex") & ((df["Case Diameter"] < 36) | (df["Case Diameter"] > 44)))]
df = df[~((df["Gender"] == "Male") & (df["Case Diameter"] > 50))]


## IQR & ZSCORE for Water Resistance
## --- IQR ---
q1 = df["Water Resistance"].quantile(0.25)
q3 = df["Water Resistance"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df[(df["Water Resistance"] < lower_bound) | (df["Water Resistance"] > upper_bound)]


df[
    (df['Water Resistance'] >= 30) |
    (df['Water Resistance'] <= 1200)
]

## IQR & ZSCORE for Weight (g)
## --- IQR ---
q1 = df["Weight (g)"].quantile(0.25)
q3 = df["Weight (g)"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df[(df["Weight (g)"] < lower_bound) | (df["Weight (g)"] > upper_bound)]

df[(df["Gender"] == "Female") & ((df["Weight (g)"] < 20) | (df["Weight (g)"] > 150))]
df = df[~((df["Gender"] == "Female") & ((df["Weight (g)"] < 20) | (df["Weight (g)"] > 150)))] # Take a look.

df[(df["Gender"] == "Male") & ((df["Weight (g)"] < 40) | (df["Weight (g)"] > 250))]
df = df[~((df["Gender"] == "Male") & ((df["Weight (g)"] < 40) | (df["Weight (g)"] > 250)))]

df[(df["Gender"] == "Unisex") & ((df["Weight (g)"] < 30) | (df["Weight (g)"] > 200))]
df = df[~((df["Gender"] == "Unisex") & ((df["Weight (g)"] < 30) | (df["Weight (g)"] > 200)))]

## IQR & ZSCORE for Price (USD)
## --- IQR ---
q1 = df["Price (USD)"].quantile(0.25)
q3 = df["Price (USD)"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df[(df["Price (USD)"] < lower_bound) | (df["Price (USD)"] > upper_bound)] # Take a look.


# Missing Values
def check_na(data, plot: bool = False):
    """
    Check for missing values and optionally plot them.
    """
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False).round(2)
    na_dataframe = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])

    print("--- Missing Data Summary ---")

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title('Missing Data Visualization')
        plt.show()

    return na_dataframe


check_na(df)

## Additional Feature
df['Additional Feature'] = df['Additional Feature'].fillna('No')

## Case Diameter
median_by_gender = df.groupby('Gender')['Case Diameter'].median()

df['Case Diameter'].fillna(
    df.groupby('Gender')['Case Diameter'].transform('median'),
    inplace=True
)

## Price (USD)
median_price_group = df.groupby(['Gender', 'Case Material', 'Brand'])['Price (USD)'].median()

df['Price (USD)'].fillna(
    df.groupby(['Gender', 'Case Material', 'Brand'])['Price (USD)'].transform('median'),
    inplace=True
)

## Strap Color
df['Strap Color'].fillna('Unknown', inplace=True)

## Mechanism
df['Mechanism'].fillna('Unknown', inplace=True)

## Weight (g)
df['Weight (g)'].fillna(
    df.groupby(['Gender', 'Case Material'])['Weight (g)'].transform('median'),
    inplace=True
)

# --- Feature Engineering ---
df['Log_Price_USD'] = np.log1p(df['Price (USD)'])

bins = [0, 500, 2500, 10000, np.inf]
labels = ['Budget', 'Mid', 'Luxury', 'Premium Luxury']


df['Price_Segment'] = pd.cut(
    df['Price (USD)'],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True
)

df = df.drop(columns = ["Price (USD)"])

df.head()

# --- Encoding ---
df_encoded = pd.get_dummies(df, drop_first=True)

# --- Model ---
X = df_encoded.drop(columns = ["Log_Price_USD"], axis = 1)
y = df_encoded["Log_Price_USD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scaling ---
numerical_cols_for_X = [
    'Case Diameter',
    'Water Resistance',
    'Warranty (Years)',
    'Weight (g)'
]

scaler = StandardScaler()

X_train[numerical_cols_for_X] = scaler.fit_transform(X_train[numerical_cols_for_X])

X_test[numerical_cols_for_X] = scaler.transform(X_test[numerical_cols_for_X])


run_models(X_train, X_test, y_train, y_test)

### XGBOOST ###
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

param_dist = {
    'subsample': uniform(loc=0.7, scale=0.3),
    'n_estimators': randint(300, 1000),
    'max_depth': randint(4, 8),
    'learning_rate': uniform(loc=0.03, scale=0.07),
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

# --- Predictions ---
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# --- Metrics ---
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}")
print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
print(f"Best Params: {search.best_params_}")

# Feature Importance

importance = best_model.get_booster().get_score(importance_type='weight')

feature_importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df.head(20),
    palette='viridis'
)
plt.title('XGBoost Feature Importance (Top 20)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Names')
plt.show()

print("\n--- Most Important Features ---")
print(feature_importance_df.head(10))