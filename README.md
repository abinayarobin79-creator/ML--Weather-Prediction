# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To develop a machine learning model using the Random Forest Algorithm for weather prediction.
2. To analyze environmental sensor data for predicting temperature, PM2.5 levels, and energy consumption.
3. To preprocess the dataset by handling missing values and extracting useful features.
4. To split the dataset into training and testing sets for model evaluation.
5. To train a Random Forest Regressor for accurate multi-output prediction.
6. To evaluate the model performance using metrics such as MAE, MSE, and R² score.
7. To visualize feature importance to understand the influence of different environmental factors.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Abinaya R
RegisterNumber:  212225230004
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\acer\Downloads\weather-station-eee-block_2024_07_13.csv")

print(df.head())

df = df.dropna()

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df = df.drop(columns=['Timestamp'])

target_cols = ['Temperature', 'PM2.5', 'Energy']

X = df.drop(columns=target_cols)
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nEvaluation Metrics:\n")

for i, col in enumerate(target_cols):
    print(f"--- {col} ---")
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")
    print()

importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()
```

## Output:


## Result:
