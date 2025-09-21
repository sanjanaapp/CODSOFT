# Titanic Survival Prediction
# Author: Sanjana
# Description: Predict survival of Titanic passengers using Logistic Regression
# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset

print("Current Working Directory:", os.getcwd())


data = pd.read_csv("titanic.csv")

print("\nFirst 5 rows of dataset:")
print(data.head())

# Step 3: Data Cleaning

drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in data.columns]
if drop_cols:
    data.drop(drop_cols, axis=1, inplace=True)

# Fill missing values (if present)
if 'Age' in data.columns:
    data['Age'].fillna(data['Age'].median(), inplace=True)
if 'Embarked' in data.columns:
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical to numeric
le = LabelEncoder()
if 'Sex' in data.columns:
    data['Sex'] = le.fit_transform(data['Sex'])
if 'Embarked' in data.columns:
    data['Embarked'] = le.fit_transform(data['Embarked'])

print("\nDataset after cleaning:")
print(data.head())


# Step 4: Split Features & Target

X = data.drop('Survived', axis=1)  # Features
y = data['Survived']               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 5: Train Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Step 6: Predictions & Evaluation

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Feature Importance (Optional Plot)
coeff_df = pd.DataFrame(model.coef_[0], X.columns, columns=['Coefficient'])
print("\nFeature Coefficients (importance):")
print(coeff_df)

# Plot coefficients
plt.figure(figsize=(8,6))
sns.barplot(x=coeff_df.index, y=coeff_df['Coefficient'])
plt.xticks(rotation=45)
plt.title("Feature Importance from Logistic Regression")
plt.show()
