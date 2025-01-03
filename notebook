# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import os

from scipy.stats import zscore

# Install LightGBM and XGBoost
!pip install lightgbm xgboost

import kagglehub

# Download latest version
path = kagglehub.dataset_download("camnugent/california-housing-prices")

print("Path to dataset files:", path)

import os

# List files in the directory
directory = '/root/.cache/kagglehub/datasets/camnugent/california-housing-prices/versions/1'
print(os.listdir(directory))

# Load the dataset
housing_data = pd.read_csv('/root/.cache/kagglehub/datasets/camnugent/california-housing-prices/versions/1/housing.csv')

# Display the first few rows of the dataset
print(housing_data.head())

# Basic information about the dataset
print(housing_data.info())

# Summary statistics of the dataset
print(housing_data.describe())

# Check for missing values
print(housing_data.isnull().sum())

# Separate numerical and categorical columns
numerical_cols = housing_data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = housing_data.select_dtypes(include=['object']).columns

# Impute missing values for numerical columns with mean
imputer_num = SimpleImputer(strategy='mean')
housing_data[numerical_cols] = imputer_num.fit_transform(housing_data[numerical_cols])

# Impute missing values for categorical columns with most frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
housing_data[categorical_cols] = imputer_cat.fit_transform(housing_data[categorical_cols])

# Verify that there are no more missing values
print(housing_data.isnull().sum())

# Feature Engineering: Creating new features
housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
housing_data['bedrooms_per_room'] = housing_data['total_bedrooms'] / housing_data['total_rooms']
housing_data['population_per_household'] = housing_data['population'] / housing_data['households']

# Adding Z-score, variance and standard deviation
numeric_features = housing_data.select_dtypes(include=[np.number]).columns.tolist()
for feature in numeric_features:
    housing_data[f'{feature}_zscore'] = zscore(housing_data[feature])
    housing_data[f'{feature}_variance'] = housing_data[feature].var()
    housing_data[f'{feature}_std'] = housing_data[feature].std()

# Display the first few rows to see the new features
print(housing_data.head())

# Feature Engineering: Creating new features
housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
housing_data['bedrooms_per_room'] = housing_data['total_bedrooms'] / housing_data['total_rooms']
housing_data['population_per_household'] = housing_data['population'] / housing_data['households']

# Adding Z-score, variance and standard deviation
numeric_features = housing_data.select_dtypes(include=[np.number]).columns.tolist()
for feature in numeric_features:
    housing_data[f'{feature}_zscore'] = zscore(housing_data[feature])
    housing_data[f'{feature}_variance'] = housing_data[feature].var()
    housing_data[f'{feature}_std'] = housing_data[feature].std()

# Display the first few rows to see the new features
print(housing_data.head())

# Feature Engineering: Creating new features
housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
housing_data['bedrooms_per_room'] = housing_data['total_bedrooms'] / housing_data['total_rooms']
housing_data['population_per_household'] = housing_data['population'] / housing_data['households']

# Adding Z-score, variance and standard deviation
numeric_features = housing_data.select_dtypes(include=[np.number]).columns.tolist()
for feature in numeric_features:
    housing_data[f'{feature}_zscore'] = zscore(housing_data[feature])
    housing_data[f'{feature}_variance'] = housing_data[feature].var()
    housing_data[f'{feature}_std'] = housing_data[feature].std()

# Display the first few rows to see the new features
print(housing_data.head())

# Feature Engineering: Creating new features
housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
housing_data['bedrooms_per_room'] = housing_data['total_bedrooms'] / housing_data['total_rooms']
housing_data['population_per_household'] = housing_data['population'] / housing_data['households']

# Adding Z-score, variance and standard deviation
numeric_features = housing_data.select_dtypes(include=[np.number]).columns.tolist()
for feature in numeric_features:
    housing_data[f'{feature}_zscore'] = zscore(housing_data[feature])
    housing_data[f'{feature}_variance'] = housing_data[feature].var()
    housing_data[f'{feature}_std'] = housing_data[feature].std()

# Display the first few rows to see the new features
print(housing_data.head())

# Feature Engineering: Creating new features
housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
housing_data['bedrooms_per_room'] = housing_data['total_bedrooms'] / housing_data['total_rooms']
housing_data['population_per_household'] = housing_data['population'] / housing_data['households']

# Adding Z-score, variance and standard deviation
numeric_features = housing_data.select_dtypes(include=[np.number]).columns.tolist()
for feature in numeric_features:
    housing_data[f'{feature}_zscore'] = zscore(housing_data[feature])
    housing_data[f'{feature}_variance'] = housing_data[feature].var()
    housing_data[f'{feature}_std'] = housing_data[feature].std()

# Display the first few rows to see the new features
print(housing_data.head())
