import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer


# --- 1. Custom Transformer ---

class GroupBasedImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.imputation_maps = {}

    def fit(self, X, y=None):
        self.imputation_maps['Case Diameter'] = X.groupby('Gender')['Case Diameter'].median()
        self.imputation_maps['Weight (g)'] = X.groupby(['Gender', 'Case Material'])['Weight (g)'].median()
        return self

    def transform(self, X):
        X_imp = X.copy()

        X_imp['Case Diameter'] = X_imp.groupby('Gender')['Case Diameter'].transform(
            lambda x: x.fillna(self.imputation_maps['Case Diameter'].loc[x.name])
        )

        X_imp['Weight (g)'] = X_imp.groupby(['Gender', 'Case Material'])['Weight (g)'].transform(
            lambda x: x.fillna(self.imputation_maps['Weight (g)'].loc[x.name[0], x.name[1]])
        )

        X_imp['Additional Feature'] = X_imp['Additional Feature'].fillna('No')
        X_imp['Strap Color'] = X_imp['Strap Color'].fillna('Unknown')
        X_imp['Mechanism'] = X_imp['Mechanism'].fillna('Unknown')

        return X_imp


# --- 2. Data Preparation and Synchronization ---

data = pd.read_csv("data/watch_price_20250910.csv")
df = data.copy()

df = df[~((df["Gender"] == "Female") & (df["Case Diameter"] > 40))]
df = df[~((df["Gender"] == "Unisex") & ((df["Case Diameter"] < 36) | (df["Case Diameter"] > 44)))]
df = df[~((df["Gender"] == "Male") & (df["Case Diameter"] > 50))]
df = df[~((df["Gender"] == "Female") & ((df["Weight (g)"] < 20) | (df["Weight (g)"] > 150)))]
df = df[~((df["Gender"] == "Male") & ((df["Weight (g)"] < 40) | (df["Weight (g)"] > 250)))]
df = df[~((df["Gender"] == "Unisex") & ((df["Weight (g)"] < 30) | (df["Weight (g)"] > 200)))]

df['Price (USD)'] = df['Price (USD)'].fillna(
    df.groupby(['Gender', 'Case Material', 'Brand'])['Price (USD)'].transform('median')
)

y_log = np.log1p(df['Price (USD)'])
X = df.drop(columns=['Price (USD)'])

X_train_raw, X_test_raw, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# --- 3. Pipeline Configuration ---

NUMERICAL_FEATURES = [
    'Case Diameter',
    'Water Resistance',
    'Warranty (Years)',
    'Weight (g)'
]
CATEGORICAL_FEATURES = X_train_raw.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num',
         Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
         NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)

OPTIMIZED_XGB_PARAMS = {
    'colsample_bytree': 0.8,
    'gamma': 0,
    'learning_rate': 0.04412034416347774,
    'max_depth': 7,
    'n_estimators': 540,
    'subsample': 0.794306794322898,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}
optimized_xgb = XGBRegressor(**OPTIMIZED_XGB_PARAMS)

FULL_PIPELINE = Pipeline(steps=[
    ('imputer_custom', GroupBasedImputer()),
    ('preprocessor', preprocessor),
    ('regressor', optimized_xgb)
])

# --- 4. Training and Recording ---

MODEL_FILENAME = 'optimized_watch_price_xgb_pipeline.joblib'

print("Starting final Production Pipeline training on full training set...")
FULL_PIPELINE.fit(X_train_raw, y_train_log)
print("Pipeline training complete. Model is ready for deployment.")

joblib.dump(FULL_PIPELINE, MODEL_FILENAME)

print(f"\n✅ SUCCESSFUL: Pipeline saved to file '{MODEL_FILENAME}'.")