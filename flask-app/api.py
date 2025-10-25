import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify, render_template
from sklearn.base import BaseEstimator, TransformerMixin


# Modelinizin bu özel dönüştürücülere ihtiyaç duyduğu için bu sınıflar kalmalıdır.
class GroupBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputation_maps = {}
        self.global_median_case_diameter = None
        self.global_median_weight_g = None

    def fit(self, X, y=None):
        X_fit = X.copy().dropna(subset=['Case Diameter', 'Weight (g)'])
        if 'Gender' in X_fit.columns:
            self.imputation_maps['Case Diameter'] = X_fit.groupby('Gender')['Case Diameter'].median()
        if 'Gender' in X_fit.columns and 'Case Material' in X_fit.columns:
            self.imputation_maps['Weight (g)'] = X_fit.groupby(['Gender', 'Case Material'])['Weight (g)'].median()
        if 'Case Diameter' in X_fit:
            self.global_median_case_diameter = X_fit['Case Diameter'].median()
        if 'Weight (g)' in X_fit:
            self.global_median_weight_g = X_fit['Weight (g)'].median()
        return self

    def transform(self, X):
        X_imp = X.copy()
        if hasattr(self, 'imputation_maps') and 'Case Diameter' in self.imputation_maps and hasattr(self,
                                                                                                    'global_median_case_diameter') and self.global_median_case_diameter is not None:
            if 'Gender' in X_imp.columns:
                def impute_case_diameter(x):
                    key = x.name
                    median_val = self.imputation_maps['Case Diameter'].get(key, self.global_median_case_diameter)
                    return x.fillna(median_val)

                X_imp['Case Diameter'] = X_imp.groupby('Gender')['Case Diameter'].transform(impute_case_diameter)

        if hasattr(self, 'imputation_maps') and 'Weight (g)' in self.imputation_maps and hasattr(self,
                                                                                                 'global_median_weight_g') and self.global_median_weight_g is not None:
            if 'Gender' in X_imp.columns and 'Case Material' in X_imp.columns:
                weight_imputer = self.imputation_maps['Weight (g)']
                global_fallback = self.global_median_weight_g

                def get_imputation_value(row):
                    key = (row['Gender'], row['Case Material'])
                    try:
                        return weight_imputer.loc[key]
                    except KeyError:
                        return global_fallback

                nan_weight_indices = X_imp[X_imp['Weight (g)'].isna()].index
                if not nan_weight_indices.empty:
                    imputed_values = X_imp.loc[nan_weight_indices].apply(get_imputation_value, axis=1)
                    X_imp.loc[nan_weight_indices, 'Weight (g)'] = imputed_values

        if 'Additional Feature' in X_imp.columns: X_imp['Additional Feature'] = X_imp['Additional Feature'].fillna('No')
        if 'Strap Color' in X_imp.columns: X_imp['Strap Color'] = X_imp['Strap Color'].fillna('Unknown')
        if 'Mechanism' in X_imp.columns: X_imp['Mechanism'] = X_imp['Mechanism'].fillna('Unknown')

        return X_imp


# --- GLOBAL VARIABLES & HELPERS ---
if os.environ.get("RUNNING_IN_DOCKER"):
    MODEL_PATH = "/models/optimized_watch_price_xgb_pipeline.joblib"
else:
    MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "optimized_watch_price_xgb_pipeline.joblib"))
PIPELINE = None


def load_model():
    global PIPELINE
    try:
        PIPELINE = joblib.load(MODEL_PATH)
        print("✅ Model Pipeline loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR loading the pipeline: {e}")
        PIPELINE = None


def get_price_segment(price: float) -> str:
    if 0 <= price < 500: return 'Budget'
    if 500 <= price < 2500: return 'Mid-Range'
    if 2500 <= price < 10000: return 'Luxury'
    return 'Premium Luxury'


# --- FLASK APP SETUP ---
app = Flask(__name__, static_folder='static', template_folder='templates')


# --- SİTE SAYFALARI ---
@app.route('/')
def home(): return render_template('home.html')


@app.route('/contents')
def contents(): return render_template('contents.html')


@app.route('/predict')
def predict_page(): return render_template('predict.html')


# --- TAHMİN API ENDPOINT'İ ---
@app.route('/predict_api', methods=['POST'])
def predict_api():
    if PIPELINE is None: return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500
    if not request.json: return jsonify({'error': 'Missing JSON data.'}), 400

    data = request.json
    REQUIRED_COLUMNS = ['Brand', 'Model', 'Gender', 'Case Diameter', 'Water Resistance', 'Case Color', 'Glass Shape',
                        'Origin', 'Case Material', 'Additional Feature', 'Strap Color', 'Warranty (Years)',
                        'Strap Material', 'Mechanism', 'Glass Type', 'Dial Color', 'Weight (g)']

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in data]
    if missing_cols: return jsonify({'error': f'Missing data for: {", ".join(missing_cols)}'}), 400

    try:
        input_df = pd.DataFrame([data], columns=REQUIRED_COLUMNS)
        numeric_cols = ['Case Diameter', 'Water Resistance', 'Warranty (Years)', 'Weight (g)']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        log_price_prediction = PIPELINE.predict(input_df)
        predicted_price_usd = np.expm1(log_price_prediction)[0]
        price = max(0, predicted_price_usd)
        segment = get_price_segment(price)

        # --- HATA ÇÖZÜMÜ: NumPy float'ı standart Python float'ına dönüştür ---
        return jsonify({
            'predicted_price': float(round(price, 2)),  # Sadece 'float()' eklendi
            'currency': 'USD',
            'segment': segment
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500


# --- UYGULAMA BAŞLATMA ---
if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', debug=True, port=5000)