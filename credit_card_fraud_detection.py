import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
# Load dataset
df = pd.read_csv("creditcard.csv")

print("Shape of dataset:", df.shape)
print(df.head())
# Drop Time column
df = df.drop(["Time"], axis=1)

# Normalize Amount
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]
# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Original class distribution:\n", y.value_counts())
print("Resampled class distribution:\n", y_resampled.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
# Confusion matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm)

