import numpy as np
import pandas as pd
from sklearn import tree
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Import the data file
input_file = "../mammographic_masses.data.txt"
df = pd.read_csv(input_file, header = 0)

# Define the column names
column_names = [
    "BI-RADS",  # Not used in prediction
    "Age",
    "Shape",
    "Margin",
    "Density",
    "Severity"
]
# Assign the column names to the DataFrame
df.columns = column_names
# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)
print(df.head())

# Check data types of each column
print("Data types of each column:")
print(df.dtypes)
# Convert relevant columns to numeric types, forcing errors to NaN
numeric_columns = ["Age", "Shape", "Margin", "Density", "Severity"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
# Display the statistical summary of the dataframe again
print("\nStatistical summary of the dataset:")
print(df.describe())
# Check for missing values in the dataset
print("\nMissing values in each column:")
print(df.isnull().sum())

# Apply MICE for imputation, in order to fill the missing data with expected-predicted values
imputer = IterativeImputer(random_state=0)
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
# Round up imputed values in nominal columns (Shape, Margin, Density)
nominal_columns = ["Shape", "Margin", "Density"]
df[nominal_columns] = df[nominal_columns].round()
# Check for missing values after imputation
print("\nMissing values in each column after imputation:")
print(df.isnull().sum())
# Display the first few rows of the updated DataFrame
print("\nFirst 5 rows of the dataset after imputation:")
print(df.head())

# Extracting feature data and target class, in order to prepare the data for the numpy's precedures
X = df[["Age", "Shape", "Margin", "Density"]].values  # Feature array (Age, Shape, Margin, Density)
y = df["Severity"].values                             # Target array (Severity)
# Feature names
feature_names = ["Age", "Shape", "Margin", "Density"]
# Check the output
print("Feature array (X):")
print(X[:5])  # Display the first 5 rows
print("\nTarget array (y):")
print(y[:5])  # Display the first 5 rows
print("\nFeature names:")
print(feature_names)

# Normalize the feature data, some models need normalized data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
# Check the output
print("Normalized feature array (X):")
print(X_normalized[:5])  # Display the first 5 rows of normalized data
