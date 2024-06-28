# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:45:56 2024

@author: Maciek
"""
#Importing libraries
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
X_train = df.iloc[0:500, 2:7]
y_train = df.iloc[0:500, 1]

# Inputs to test
X_test = df.iloc[501:550, 2:7]
y_test = df.iloc[501:550, 1]


# Initialize and train the logistic regression model
logreg = LogisticRegression(max_iter=10000, random_state=42)
logreg.fit(X_train, y_train)

# Get predicted probabilities
y_pred_proba = logreg.predict_proba(X_test)

# Example raw input for prediction
#raw_list = [[13.21, 28.06, 84.88, 538.4, 0.08671]]
#Percentage answer should be 0.43


#Inputs from Console for prediction
print('Predict the chance for breast cancer to be maligant')
new_prediction = [[]]
radius_mean = float(input('Write mean radius:'))
new_prediction[0].append(radius_mean)
texture_mean = float(input('Write texture:'))
new_prediction[0].append(texture_mean)
perimeter_mean = float(input('Write perimeter:'))
new_prediction[0].append(perimeter_mean)
area_mean = float(input('Write mean area:'))
new_prediction[0].append(area_mean)
smoothness_mean = float(input('Write smoothness rate:'))
new_prediction[0].append(smoothness_mean)
input_y_new_pred = logreg.predict_proba(new_prediction)


print(f'Probability for Maligant: {input_y_new_pred[0,0]:.2f}')

print('done')
input('Press any key to close')
