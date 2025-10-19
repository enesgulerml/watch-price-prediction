# Watch Price Prediction: An AI-Driven Valuation Model

A comprehensive, end-to-end machine learning project that predicts the market value of luxury watches based on their specifications. This project features a complete pipeline, from data exploration and feature engineering to model training and deployment via a responsive web application built with Flask.

## ✨ Key Features

-   **Accurate Price Prediction:** Utilizes an optimized XGBoost regression model to estimate watch prices with high accuracy (R² of 0.99).
-   **Interactive Web Interface:** A user-friendly and responsive UI built with Flask, HTML, CSS, and JavaScript, allowing users to input watch features and get instant predictions.
-   **Comprehensive Feature Set:** The model considers 17 different features, including brand, model, case material, and more, based on a real-world dataset.
-   **Dynamic Content:** An "All Brands" page dynamically generated with JavaScript, showcasing a wide variety of watches.
-   **Light & Dark Mode:** A modern, user-selectable theme for comfortable viewing in any lighting condition.

## 🚀 Tech Stack

This project was built using a combination of data science and web development technologies:

-   **Backend:** Python, Flask
-   **Frontend:** HTML5, CSS3, JavaScript
-   **ML/Data Science:** Scikit-learn, Pandas, NumPy, XGBoost
-   **Model Management:** Joblib
-   **Development & IDE:** PyCharm

## 📂 Project Structure

The project is organized with a clear separation of concerns, making it scalable and maintainable.

ML-Resume-Regression/ ├── data/ │ └── watch_price_20250910.csv ├── flask-app/ │ ├── api.py │ ├── static/ │ │ ├── css/ │ │ ├── js/ │ │ └── images/ │ └── templates/ │ ├── home.html │ ├── contents.html │ └── predict.html ├── models/ │ └── optimized_watch_price_xgb_pipeline.joblib ├── notebooks/ │ ├── Watch Price Prediction - EDA.ipynb │ ├── Watch Price Prediction - FE.ipynb │ └── Watch Price Prediction - Model.ipynb ├── .gitignore ├── README.md └── requirements.txt

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd ML-Resume-Regression
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    cd flask-app
    python api.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

## 📈 Model Performance

The final XGBoost model was trained on a comprehensive dataset and evaluated based on the following metrics:

-   **R² Score:** **0.99**
-   **Normalized RMSE:** Approximately **8%** (Coefficient of Variation of the RMSE)

## 👤 Author

**Enes Güler**

-   **GitHub:** [enesgulerml](https://github.com/enesgulerml)
-   **LinkedIn:** [Enes Güler](https://www.linkedin.com/in/enes-güler-8ab8a7346)
-   **Medium:** [@ml.enesguler](https://medium.com/@ml.enesguler)

## 📄 License

This project is licensed under the MIT License.