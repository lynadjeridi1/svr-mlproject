import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset.csv")

# Basic EDA
print(df.head())
print(df.describe())

sns.heatmap(df.corr(), annot=True)
plt.show()

# Split features and target
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# SVR with hyperparameter tuning
param_grid = {
    "C": [0.1, 1, 10],
    "epsilon": [0.1, 0.2, 0.5],
    "kernel": ["rbf", "linear"]
}

svr = GridSearchCV(SVR(), param_grid, cv=3)
svr.fit(X_train, y_train)

best_svr = svr.best_estimator_

# Regression evaluation
svr_preds = best_svr.predict(X_test)
print("SVR MSE:", mean_squared_error(y_test, svr_preds))

# Logistic regression baseline
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_preds = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, log_preds))
print("Precision:", precision_score(y_test, log_preds, average='weighted'))
print("Recall:", recall_score(y_test, log_preds, average='weighted'))
print("F1 Score:", f1_score(y_test, log_preds, average='weighted'))
