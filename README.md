# Predicting-California-Housing-Prices-Using-Machine-Learning-Models
Overview
This project focuses on analyzing and predicting housing prices in California using machine learning techniques. The dataset used is the California Housing Prices dataset, which includes various features such as median income, total rooms, total bedrooms, population, households, and more. The goal is to preprocess the data, perform feature engineering, and build several regression models to accurately predict the median house values.

Steps Involved
Data Collection: The dataset is downloaded from Kaggle using the Kaggle API.
Data Preprocessing: This step involves handling missing values, converting categorical variables into numerical ones, and standardizing the features.
Feature Engineering: New features are created to improve the model's performance, such as rooms_per_household, bedrooms_per_room, and population_per_household.
Model Training and Evaluation: Various machine learning models, including Linear Regression, Random Forest, and Gradient Boosting, are trained and evaluated using techniques like RandomizedSearchCV for hyperparameter tuning.
Model Performance Visualization: The performance of the models is visualized using scatter plots and bar charts to show feature importance.

Detailed Steps
1. Data Collection
The dataset is downloaded from Kaggle using the kagglehub library. The data is then loaded into a pandas DataFrame for further processing.
2. Data Preprocessing
Handling missing values and converting categorical data into dummy variables.
3. Feature Engineering
Creating new features to enhance model performance.
4. Model Training and Evaluation
Training and evaluating different regression models.


Conclusion
This project demonstrates the process of analyzing and predicting housing prices using machine learning. By processing the data, performing feature engineering, and evaluating multiple models, we can identify the best model for predicting housing prices in California. The visualizations provide insights into the model's performance and the importance of different features.
