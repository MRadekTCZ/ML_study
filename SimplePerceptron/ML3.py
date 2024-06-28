# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:52:59 2024

@author: Maciek
"""
#Download libraries
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
# Load data
df = pd.read_csv('breast_cancer.csv')


#Inputs to train
X_train = df.iloc[0:500, 2:32]
#Output of model to train
y_train = df.iloc[0:500, 1]

#Inputs to test
X_test = df.iloc[501:550, 2:32]
#Inputs to test
y_test = df.iloc[501:550, 1] 

# Initialize and train the perceptron model
perceptron = Perceptron(max_iter=10000, eta0=0.1, random_state=42)
perceptron.fit(X_train, y_train)
joblib.dump(perceptron, 'perceptron_model.pkl')
# Make predictions
y_pred = perceptron.predict(X_test)

#Example Raw input for prediction
raw_list = [[13.61, 24.98, 88.05, 582.7, 0.09488, 0.08511, 0.08625, 0.04489,
             0.1609, 0.05871, 0.4565, 1.29, 2.861, 43.14, 0.005872, 0.01488, 
             0.02647, 0.009921, 0.01465, 0.002355, 16.99, 35.27, 108.6, 906.5, 
             0.1265, 0.1943, 0.3169, 0.1184, 0.2651, 0.07397]]

one_y_pred = perceptron.predict(raw_list)
print(one_y_pred)
# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')



