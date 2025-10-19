import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# --- 1. Custom Transformers ---

class GroupBasedImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using group-based medians for Case Diameter and Weight (g), with global fallback."""

    def __init__(self):
        self.imputation_maps = {}
        self.global_median_case_diameter = None
        self.global_median_weight_g = None

    def fit(self, X, y=None):
        """Calculates the median values and global fallbacks."""
        X_fit = X.copy().dropna(subset=['Case Diameter', 'Weight (g)'])

        self.imputation_maps['Case Diameter'] = X_fit.groupby('Gender')['Case Diameter'].median()
        self.imputation_maps['Weight (g)'] = X_fit.groupby(['Gender', 'Case Material'])['Weight (g)'].median()

        self.global_median_case_diameter = X_fit['Case Diameter'].median()
        self.global_median_weight_g = X_fit['Weight (g)'].median()

        return self

    def transform(self, X):
        """Applies the imputation using the safer map-based lookup with global fallback."""
        X_imp = X.copy()

        # --- Case Diameter Imputation ---
        def impute_case_diameter(x):
            key = x.name
            # Safer lookup using .get()
            median_val = self.imputation_maps['Case Diameter'].get(key, self.global_median_case_diameter)
            return x.fillna(median_val)

        X_imp['Case Diameter'] = X_imp.groupby('Gender')['Case Diameter'].transform(impute_case_diameter)

        # --- Weight (g) Imputation (KeyError Çözümü) ---
        weight_imputer = self.imputation_maps['Weight (g)']
        global_fallback = self.global_median_weight_g

        # Define fill function for apply
        def get_imputation_value(row):
            key = (row['Gender'], row['Case Material'])

            try:
                # Direct MultiIndex lookup
                return weight_imputer.loc[key]
            except KeyError:
                # Fallback to global median if combination is unseen
                return global_fallback

        nan_weight_indices = X_imp[X_imp['Weight (g)'].isna()].index

        if not nan_weight_indices.empty:
            imputed_values = X_imp.loc[nan_weight_indices].apply(get_imputation_value, axis=1)
            X_imp.loc[nan_weight_indices, 'Weight (g)'] = imputed_values

        # Simple fill for remaining categorical NaNs
        X_imp['Additional Feature'] = X_imp['Additional Feature'].fillna('No')
        X_imp['Strap Color'] = X_imp['Strap Color'].fillna('Unknown')
        X_imp['Mechanism'] = X_imp['Mechanism'].fillna('Unknown')

        return X_imp


# --- 2. Configuration and Constants (ALL Data Values Included) ---

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME = "optimized_watch_price_xgb_pipeline.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

PRICE_SEGMENTS = {
    'Budget': (0, 500),
    'Mid': (500, 2500),
    'Luxury': (2500, 10000),
    'Premium Luxury': (10000, np.inf)
}

# --- COMPLETE DATA-DRIVEN OPTION LISTS ---
BRAND_OPTIONS = ['Audemars Piguet', 'Breitling', 'Bulova', 'Cartier', 'Casio', 'Casio G-Shock', 'Citizen',
                 'Daniel Wellington', 'Fossil', 'Girard-Perregaux', 'Hamilton', 'Hublot', 'Invicta', 'Longines',
                 'Maurice Lacroix', 'Michael Kors', 'Movado', 'Orient', 'Panerai', 'Patek Philippe', 'Rado', 'Rolex',
                 'Seiko', 'Skagen', 'Swatch', 'Tag Heuer', 'Timex', 'Tissot', 'Vacheron Constantin']
GENDER_OPTIONS = ['Female', 'Male', 'Unisex']
CASE_MATERIAL_OPTIONS = ['Alloy', 'Steel', 'Titanium']
ADDITIONAL_FEATURES_OPTIONS = ['Alarm', 'Calendar', 'Chronograph', 'Luminous', 'No', 'None']
STRAP_COLOR_OPTIONS = ['Black', 'Blue', 'Brown', 'Gold', 'Silver', 'Unknown', 'White']
MECHANISM_OPTIONS = ['Automatic', 'Mechanical', 'Quartz', 'Unknown']

# Placeholder/Less Important Feature Options (For mandatory UI completion)
MODEL_OPTIONS = ['1966', '5 Sports', 'Aerospace', 'Aikon', 'Allied Coastline', 'American Classic', 'Ancher', 'Aquanaut',
                 'Aquaracer', 'Astron', 'Autavia', 'Automatic', 'Avenger', 'Aviator', 'Baignoire', 'Ballon Bleu',
                 'Bambino', 'Big Bang', 'Big Bold', 'Blue Angels', 'Bold', 'Bolt', 'Bradshaw', 'Bridges', 'Broadway',
                 'Calatrava', 'Captain Cook', 'Carrera', "Cat's Eye", 'Centrix', 'Ceramica', 'Chrono', 'Chrono Hawk',
                 'Chronomaster', 'Chronomat', 'Classic', 'Classic Bristol', 'Classic Fusion', 'Classic Sheffield',
                 'Classic St Mawes', 'Clé', 'Code 11.59', 'Collection', 'Complications', 'Concept', 'Connect',
                 'Connected', 'Conquest', 'Curv', 'DW-5600', 'Darci', 'Databank', 'Datejust', 'Daytona', 'Defender',
                 'DolceVita', 'Drive de Cartier', 'Due', 'Eco-Drive', 'Edge', 'Edifice', 'Edward Piguet', 'Egiziano',
                 'Eliros', 'Endurance Pro', 'Esperanza', 'Everytime', 'Expedition', 'Explorer', 'FB-01', 'Fairfield',
                 'Falster', 'Fiaba', 'Fiftysix', 'Flagship', 'Florence', 'Formula 1', 'Frogman', 'G-Steel', 'GA-2100',
                 'GMT', 'GMT-Master II', 'Garrett', 'Gent', 'Gentleman', 'Golden Ellipse', 'Golden Horse', 'Gondolo',
                 'Grand Complications', 'Grant', 'Gravitymaster', 'Hagen', 'Harmony', 'Heritage', 'Heritage Visodate',
                 'Historiques', 'Holst', 'HydroConquest', 'HyperChrome', 'Iconic Link', 'Integral', 'Ironman', 'Irony',
                 'Jacqueline', 'Jazzmaster', 'Jorn', 'Jules Audemars', 'Kamasu', 'Khaki Aviation', 'Khaki Field',
                 'Khaki Navy', 'Kinetic', 'Kristoffer', 'Laureato', 'Le Locle', 'Lennox', 'Les Classiques', 'Lexington',
                 'Lineage', 'Link', 'Lukia', 'Luminor', 'Lupah', 'MP Collection', 'MT-G', 'Machine', 'Mako II', 'Malte',
                 'Mare Nostrum', 'Marine Star', 'Marlin', 'Master Collection', 'Masterpiece', 'Melbye', 'Millenary',
                 'Millennia', 'Miros', 'Monaco', 'Monarch', 'Mudmaster', 'Museum Classic', 'Nautilus', 'Navi Harbor',
                 'Navitimer', 'Neutra', 'New Gent', 'Oceanus', 'Originals', 'Overseas', 'Oyster Perpetual', 'PRX',
                 'Panthère', 'Parker', 'Pasha', 'Patrimony', 'Petite Evergold', 'Petite Melrose', 'Petite Sterling',
                 'Phase de Lune', 'Pontos', 'Precisionist', 'Premier', 'Presage', 'Pro Diver', 'Pro Trek', 'Promaster',
                 'Prospex', 'Pyper', 'Quadro', 'Quartz', 'Radiomir', 'Rangeman', 'Ray II', 'Reserve', 'Royal Oak',
                 'Royal Oak Offshore', 'Rubaiyat', 'Runway', 'Russian Diver', 'SE Pilot', 'Sang Bleu', 'Santos',
                 'Satellite Wave', 'Scarlette', 'Sea Hawk', 'Seastar', 'Signatur', 'Sistem51', 'Skin', 'Sky-Dweller',
                 'Slim Runway', 'Spirit', 'Spirit of Big Bang', 'Spirit of Liberty', 'Square Bang', 'Star', 'Subaqua',
                 'Submariner', 'Submersible', 'Sun and Moon', 'Superocean', 'Surveyor', 'Symphonette', 'T-Touch',
                 'Tank', 'Titanium', 'Top Time', 'Tourbillon', 'Townsman', 'Tradition', 'Traditionnelle', 'Traveller',
                 'True Square', 'Twenty~4', 'Ultra Slim', 'Unico', 'Velatura', 'Venom', 'Ventura', 'Vintage',
                 'Vintage 1945', 'Vizio', 'Waterbury', 'Weekender', 'Yacht-Master', 'Égérie']
CASE_COLOR_OPTIONS = ['Black', 'Blue', 'Gold', 'Rose Gold', 'Silver', 'White']
GLASS_SHAPE_OPTIONS = ['Domed', 'Flat', 'Round']
ORIGIN_OPTIONS = ['CH', 'CN', 'DE', 'DK', 'FR', 'IT', 'JP', 'KR', 'SE', 'US']
STRAP_MATERIAL_OPTIONS = ['Leather', 'Metal', 'Nylon', 'Silicone']
GLASS_TYPE_OPTIONS = ['Acrylic', 'Mineral', 'Sapphire']
DIAL_COLOR_OPTIONS = ['Black', 'Blue', 'Gold', 'Silver', 'White']
WATER_RESISTANCE_OPTIONS = [3, 5, 10, 20, 50, 100, 200]

# Define all 16 columns expected by the trained pipeline
REQUIRED_INPUT_COLUMNS = [
    'Brand', 'Model', 'Gender', 'Case Diameter', 'Water Resistance',
    'Case Color', 'Glass Shape', 'Origin', 'Case Material',
    'Additional Feature', 'Strap Color', 'Warranty (Years)',
    'Strap Material', 'Mechanism', 'Glass Type', 'Dial Color',
    'Weight (g)'
]


# --- 3. Core Functions ---

@st.cache_resource
def load_model(path: str):
    """Loads the serialized model pipeline safely."""
    try:
        pipeline = joblib.load(path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading the model pipeline. Check custom class definitions in app.py: {e}")
        st.stop()


def get_price_segment(price: float) -> str:
    """Calculates the price segment based on the predicted price."""
    for segment, (lower, upper) in PRICE_SEGMENTS.items():
        if lower <= price < upper:
            return segment
    return "Undefined"


def predict_price(pipeline, input_data: dict) -> float:
    """
    Makes a prediction using the pipeline and returns the inverse transformed price.
    Fills all 16 required columns.
    """

    # 1. Prepare Full Input Data Dictionary
    full_data = {}

    # Fill data from user inputs
    for key, value in input_data.items():
        full_data[key] = [value]

    # Fill placeholder columns (None needed since all 16 features are now in the UI)

    # 2. Create DataFrame with the required column structure/order
    input_df = pd.DataFrame(full_data, columns=REQUIRED_INPUT_COLUMNS)

    # 3. Make Prediction
    log_price_prediction = pipeline.predict(input_df)

    # Inverse transform
    predicted_price_usd = np.expm1(log_price_prediction)[0]

    return max(0, predicted_price_usd)


# --- 4. Streamlit Application Layout ---

st.set_page_config(
    page_title="Professional Watch Price Estimator",
    layout="wide",
    initial_sidebar_state="auto"
)

# Load the model once
full_pipeline = load_model(MODEL_PATH)

st.title("⌚️ Optimized Watch Price Estimator (XGBoost)")
st.markdown("""
    This application uses an **Optimized XGBoost Regression Pipeline** ($R^2$: **0.9963**) 
    to estimate the price of a watch based on **all** its features.
""")

st.divider()

# --- User Input Form (All 16 Features) ---

st.header("Input Watch Specifications (All Features)")

# Use a form for better input handling if necessary, but columns are sufficient here
col1, col2, col3 = st.columns(3)

# Column 1: Core Identification
with col1:
    st.subheader("Core Identity")

    brand = st.selectbox("Brand", options=BRAND_OPTIONS)
    model = st.selectbox("Model Name", options=MODEL_OPTIONS)
    gender = st.selectbox("Gender", options=GENDER_OPTIONS)
    origin = st.selectbox("Origin (Country Code)", options=ORIGIN_OPTIONS)

# Column 2: Materials & Design
with col2:
    st.subheader("Materials & Build")

    case_material = st.selectbox("Case Material", options=CASE_MATERIAL_OPTIONS)
    case_color = st.selectbox("Case Color", options=CASE_COLOR_OPTIONS)
    dial_color = st.selectbox("Dial Color", options=DIAL_COLOR_OPTIONS)
    strap_material = st.selectbox("Strap Material", options=STRAP_MATERIAL_OPTIONS)
    strap_color = st.selectbox("Strap Color", options=STRAP_COLOR_OPTIONS)

# Column 3: Technical Specifications
with col3:
    st.subheader("Technical Specs")

    mechanism = st.selectbox("Mechanism", options=MECHANISM_OPTIONS)
    glass_type = st.selectbox("Glass Type", options=GLASS_TYPE_OPTIONS)
    glass_shape = st.selectbox("Glass Shape", options=GLASS_SHAPE_OPTIONS)
    water_resistance = st.selectbox("Water Resistance (m)", options=WATER_RESISTANCE_OPTIONS)

    # Numerical Inputs
    case_diameter = st.number_input("Case Diameter (mm)", min_value=28.0, max_value=70.0, value=39.1, step=0.1,
                                    format="%.1f")
    weight_g = st.number_input("Weight (g)", min_value=46.7, max_value=500.0, value=72.7, step=1.0)
    warranty = st.slider("Warranty (Years)", min_value=1.0, max_value=3.0, value=2.0, step=0.5)

# Full Width Input
additional_feature = st.selectbox("Additional Feature", options=ADDITIONAL_FEATURES_OPTIONS)

st.divider()

# --- Prediction Logic ---

if st.button("Estimate Price", type="primary"):

    # 1. Prepare Input Data Dictionary (All 16 features from UI)
    input_data = {
        'Brand': brand,
        'Model': model,
        'Gender': gender,
        'Case Diameter': case_diameter,
        'Water Resistance': float(water_resistance),
        'Case Color': case_color,
        'Glass Shape': glass_shape,
        'Origin': origin,
        'Case Material': case_material,
        'Additional Feature': additional_feature,
        'Strap Color': strap_color,
        'Warranty (Years)': warranty,
        'Strap Material': strap_material,
        'Mechanism': mechanism,
        'Glass Type': glass_type,
        'Dial Color': dial_color,
        'Weight (g)': weight_g,
    }

    # 2. Make Prediction
    with st.spinner('Calculating optimized price...'):
        predicted_price = predict_price(full_pipeline, input_data)

    # 3. Get Segment and Display Results
    segment = get_price_segment(predicted_price)

    st.header("2. Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.success("✅ Estimated Price (USD)")
        st.metric(
            label="Predicted Price",
            value=f"${predicted_price:,.2f}",
            delta=f"Segment: {segment}"
        )
        st.caption("Error Margin (RMSE) on log-price: 0.0745.")

    with result_col2:
        st.info("ℹ️ Price Segmentation")

        # Segment display logic
        if segment == 'Premium Luxury':
            st.markdown(f"**This watch falls into the {segment} category.** 👑")
        elif segment == 'Luxury':
            st.markdown(f"**This watch falls into the {segment} category.** 💎")
        elif segment == 'Mid':
            st.markdown(f"**This watch falls into the {segment} category.** ✨")
        else:
            st.markdown(f"**This watch falls into the {segment} category.** 🏷️")