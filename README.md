# Predicting-California-Housing-Prices-Using-Machine-Learning-Models
Overview
This project aims to analyze and predict housing prices in California using various machine learning models. This project leverages the California Housing Prices dataset from Kaggle and utilizes advanced regression techniques to provide accurate predictions.

Table of Contents
Introduction
Dataset
Installation
Data Preprocessing
Feature Engineering
Model Training and Evaluation
Results
Conclusion
Acknowledgments

Introduction
The primary objective of this project is to predict the median house values in California based on various features available in the dataset. This project involves data preprocessing, feature engineering, model training, evaluation, and visualization of results. The models used in this project include Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, LightGBM, and XGBoost.

Dataset
The dataset used in this project is sourced from Kaggle:

California Housing Prices Dataset
The dataset contains the following columns:

longitude: Longitude coordinate
latitude: Latitude coordinate
housing_median_age: Median age of the house
total_rooms: Total number of rooms
total_bedrooms: Total number of bedrooms
population: Population in the area
households: Number of households
median_income: Median income of the households
median_house_value: Median house value

Data Preprocessing
The data preprocessing steps include:

Loading the Dataset: The dataset is loaded from the Kaggle API using the KaggleHub library.
Exploratory Data Analysis (EDA): Basic statistics, missing values, and data distribution are analyzed.
Handling Missing Values: Missing values are imputed using the mean for numerical columns and the most frequent value for categorical columns.
Data Splitting: The dataset is split into training and testing sets.
Feature Engineering
Feature engineering involves creating new features to enhance model performance. The following new features are created:

rooms_per_household: Total rooms divided by the number of households.
bedrooms_per_room: Total bedrooms divided by total rooms.
population_per_household: Population divided by the number of households.
zscore: Z-score normalization for numerical features.
variance: Variance of numerical features.
std: Standard deviation of numerical features.
Model Training and Evaluation
The following models are trained and evaluated:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
LightGBM
XGBoost
Hyperparameter tuning is performed using RandomizedSearchCV with cross-validation to find the best parameters for each model. The models are evaluated based on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics.

Results
The results of the model evaluations are visualized using scatter plots and feature importance plots. The best-performing model is identified based on the evaluation metrics.

Conclusion
This project demonstrates my ability to preprocess data, engineer new features, and apply advanced machine learning models to predict housing prices. The use of various regression techniques and hyperparameter tuning highlights my understanding and practical application of these methods. This project has improved my skills in data analysis, feature engineering, and model evaluation.

Acknowledgments
I would like to thank Kaggle for providing the dataset and the open-source community for their valuable resources and libraries.

By following this README file, you can reproduce the analysis and prediction of housing prices in California. The project showcases the practical application of machine learning techniques and demonstrates the skills acquired during its development.
