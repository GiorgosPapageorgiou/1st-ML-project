import pandas as pd
import numpy as np
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(input_file):
    """
    Loads the dataset, replaces missing values, and imputes data using MICE strategy.
    """
    df = pd.read_csv(input_file, header=None)
    # Define column names
    column_names = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
    df.columns = column_names
    # Replace "?" with NaN
    df.replace("?", np.nan, inplace=True)
    # Convert relevant columns to numeric types
    numeric_columns = ["Age", "Shape", "Margin", "Density", "Severity"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    # Apply MICE imputation, in order to fill the table with expected-predicted values, instead of missing values
    imputer = IterativeImputer(random_state=0)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def normalize_data(df):
    """
    Normalizes the feature data using StandardScaler.
    """
    X = df[["Age", "Shape", "Margin", "Density"]].values
    y = df["Severity"].values
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, y
