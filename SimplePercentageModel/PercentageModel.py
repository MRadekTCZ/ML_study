# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:38:06 2024

@author: Maciek
"""

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
# Load data
df = pd.read_csv('breast_cancer.csv')

# Inputs to train
X_train = df.iloc[0:500, 2:32]
y_train = df.iloc[0:500, 1]

# Inputs to test
X_test = df.iloc[501:550, 2:32]
y_test = df.iloc[501:550, 1]

# Initialize and train the perceptron model

#0.0146


# Initialize and train the logistic regression model
logreg = LogisticRegression(max_iter=10000, random_state=42)
logreg.fit(X_train, y_train)

# Save the model
joblib.dump(logreg, 'logistic_regression_model.pkl')

# Make predictions
y_pred = logreg.predict(X_test)

# Get predicted probabilities
y_pred_proba = logreg.predict_proba(X_test)

# Example raw input for prediction
raw_list = [[13.61, 2, 88.05, 582.7, 0.09488, 0.08511, 0.08625, 0.04489,
             0.1609, 0.05871, 0.4565, 1.29, 2.861, 1, 0.005872, 0.01488, 
             0.02647, 0.009921, 10, 0.002355, 16.99,     1, 108.6, 906.5, 
             0.1265, 0.1943, 0.3169, 0.1184, 0.2651, 0.07397]]

one_y_pred_proba = logreg.predict_proba(raw_list)
# Print the probability for the second class (Benign)
print(f'Probability for Maligant: {one_y_pred_proba[0, 0]:.2f}')

# Evaluate model performance (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
