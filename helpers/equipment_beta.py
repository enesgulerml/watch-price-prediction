import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ----- EDA -----
def check_data(data, head=5, unique_threshold=20, target=None, plot=False):
    """
    Perform exploratory data analysis (EDA) on a pandas DataFrame.

    This function provides a quick overview of the dataset by printing
    key statistics and summaries, including shape, head/tail, data types,
    memory usage, missing values, descriptive statistics, unique values,
    quantiles, correlation matrix, and optional target analysis.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to analyze.
    head : int, optional, default=5
        Number of rows to display for head and tail.
    unique_threshold : int, optional, default=20
        Maximum number of unique values to display. If a column has more
        than this number, the values are skipped but the count is shown.
    target : str, optional, default=None
        Name of the target column. If provided and exists in the DataFrame,
        the function will display distribution of the target and mean values
        of numeric features grouped by the target.
    plot : bool, optional, default=False
        If True, generates visualizations including:
            - Correlation heatmap of numerical features
            - Distribution plots of numerical features
            - Target distribution plots (if `target` is specified)

    Returns
    -------
    None
        Prints summaries and statistics directly to stdout.
    """
    print("=== CHECKING DATA ===")
    print("----- SHAPE -----")
    print(f"Rows: {data.shape[0]:,}"
          f"\nColumns: {data.shape[1]}"
          f"\nFeatures: {data.columns}")

    print("\n----- HEAD & TAIL -----")
    print("First head rows:")
    print(data.head(head))
    print("\nLast head rows:")
    print(data.tail(head))

    print("\n----- DATA TYPES -----")
    print("\nData types count:")
    print(data.dtypes.value_counts())
    print("\nData Types:")
    print(data.dtypes)

    print("\nMEMORY USAGE")
    mem = data.memory_usage(deep=True) / 1024**2
    print(mem)
    print(f"\nTotal memory usage: {mem.sum():.2f} MB")

    print("\n----- MISSING VALUES -----")
    na_df = pd.DataFrame({
        "missing_count" : data.isnull().sum(),
        "missing_ratio" : data.isnull().mean()
    })
    print(na_df[na_df["missing_count"] >0])

    print("\n----- DESCRIBE -----")
    print("\n Describe Numeric:")
    print(data.describe().T)
    print("\n Describe All:")
    print(data.describe(include="all").T.head(head))

    print("\n----- UNIQUE VALUES -----")
    for col in data.columns:
        number_unique = data[col].nunique()
        if number_unique <= unique_threshold:
            print(f"{col}: {number_unique} -> {data[col].nunique()}")
        else:
            print(f"{col}: {number_unique} unique values (>{unique_threshold}, skipped listing)")

    print("\n----- QUANTILES -----")
    quantiles = [0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]
    q_df = data.select_dtypes(include=np.number).quantile(quantiles).T
    print(q_df)

    print("\n----- CORRELATION MATRIX -----")
    corr = data.select_dtypes(include=np.number).corr()
    print(corr)

    if target and target in data.columns:
        print(f"\n----- TARGET ANALYSIS ({target}) -----")
        print("\nTarget distribution:")
        print(data[target].value_counts(normalize=True))

        print("\nNumeric features vs target mean:")
        for col in data.select_dtypes(include=[np.number]):
            if col != target:
                print(f"\n{col}:")
                print(data.groupby(target)[col].mean())

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Numeric distributions
        num_cols = data.select_dtypes(include="number").columns
        data[num_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle("Numeric Feature Distributions")
        plt.tight_layout()
        plt.show()

        # Correlation heatmap
        if len(num_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(data[num_cols].corr(), cmap="coolwarm", annot=True)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()

        # Target plots
        if target and target in data.columns:
            if data[target].dtype == "object":
                sns.countplot(x=data[target])
                plt.title(f"Target Distribution: {target}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                sns.histplot(data[target], kde=True)
                plt.title(f"Target Distribution: {target}")
                plt.tight_layout()
                plt.show()


# ----- NA -----

class NaPackage:
    """
    A class for checking and visualizing missing data in a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        self.data = data.copy()

    def check_na(self, plot: bool = False):
        """
        Check for missing values and optionally plot them.
        """
        na_columns = [col for col in self.data.columns if self.data[col].isnull().sum() > 0]
        n_miss = self.data[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (self.data[na_columns].isnull().sum() / self.data.shape[0] * 100).sort_values(ascending=False).round(2)
        na_dataframe = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])

        print("--- Missing Data Summary ---")

        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.data.isnull(), cbar=False, cmap="viridis")
            plt.xticks(rotation=45, ha="right")
            plt.title('Missing Data Visualization')
            plt.show()

        return na_dataframe

    def fill_na(self, variable: str = None, method: str ="mean"):
        """
        This function helps to handle with missing values.

        Parameters
        ---------
        self : pandas.DataFrame
        variable : string, value that you want to fill.
        method : string, method that you want to use. It can be 'mean', 'median', 'mode', and 'drop'. Default is 'mean'.

        Will added "custom" to method.
        """

        if variable not in self.data.columns:
            raise KeyError(f"Column '{variable}' not found in the DataFrame.")

        if method == "mean":
            print(f"\nACTION: Filling '{variable}' using the '{method}'.")
            self.data[variable] = self.data[variable].fillna(self.data[variable].mean())

        elif method == "median":
            print(f"\nACTION: Filling '{variable}' using the '{method}'.")
            self.data[variable] = self.data[variable].fillna(self.data[variable].median())

        elif method == "mode":
            print(f"\nACTION: Filling '{variable}' using the '{method}'.")
            self.data[variable] = self.data[variable].fillna(self.data[variable].mode()[0])

        elif method == "drop":
            print(f"\nACTION: DROPPING rows with NaN values in the '{variable}' column.")
            self.data.dropna(subset=[variable], inplace=True)

        else:
            raise ValueError(f"Invalid mode '{method}'. Please use 'mean', 'median', 'mode', or 'drop'.")

        return self.data

    def fill_na_by_category(self, group_by_column, target_column, method="mean", reference_category=None):
        """
        Fill missing values in a numeric column based on category statistics or a custom function.

        Parameters
        ----------
        group_by_column : str
            Categorical column for grouping.
        target_column : str
            Numeric column with missing values.
        method : str, optional
            Fill method: 'mean', 'median', 'mode', or 'custom'.
        reference_category : optional
            Specific category to fill. If None, applies to all categories.

        Returns
        -------
        pd.DataFrame
            DataFrame with filled missing values.
        """

        if reference_category is not None:
            categories = [reference_category]
        else:
            categories = self.data[group_by_column].dropna().unique()

        for cat in categories:
            mask = (self.data[group_by_column] == cat) & (self.data[target_column].isnull())

            if method == "mean":
                group_stats = self.data.groupby(group_by_column)[target_column].agg(method)
                if cat not in group_stats.index:
                    continue
                fill_value = group_stats[cat]

            elif method == "median":
                group_stats = self.data.groupby(group_by_column)[target_column].agg(method)
                if cat not in group_stats.index:
                    continue
                fill_value = group_stats[cat]

            elif method == "mode":
                group_stats = self.data.groupby(group_by_column)[target_column].agg(method)
                if cat not in group_stats.index:
                    continue
                fill_value = group_stats[cat]

            else:
                raise ValueError("Invalid method or missing custom function.")

            self.data.loc[mask, target_column] = fill_value

        print(
            f"Completed filling missing values in '{target_column}' using '{method}' for categories: {', '.join(categories)}.")
        return self.data

# ----- OUTLIER -----



# ----- ENCODE -----


# ----- MODEL -----
def run_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "KNN" : KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1),
        "LightGBM": LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # R2 Scores
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # RMSE Scores
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        if r2_train - r2_test > 0.1:
            status = "Overfit"
        elif r2_train < 0.5 and r2_test < 0.5:
            status = "Underfit"
        else:
            status = "Good Fit"

        results.append([name, r2_train, r2_test, rmse_train, rmse_test, status])

    results_df = pd.DataFrame(results, columns=["Model", "Train R2", "Test R2", "Train RMSE", "Test RMSE", "Status"])

    results_df_formatted = results_df.copy()
    results_df_formatted["Train R2"] = results_df_formatted["Train R2"].map("{:.4f}".format)
    results_df_formatted["Test R2"] = results_df_formatted["Test R2"].map("{:.4f}".format)
    results_df_formatted["Train RMSE"] = results_df_formatted["Train RMSE"].map("{:.2f}".format)
    results_df_formatted["Test RMSE"] = results_df_formatted["Test RMSE"].map("{:.2f}".format)

    print(results_df_formatted.to_string(index=False))
## This thing is good, but 2_train - r2_test > 0.1: this thing is optional.
# You can change the limit 0.1 to any number you want.



# ----- USAGE PART -----
