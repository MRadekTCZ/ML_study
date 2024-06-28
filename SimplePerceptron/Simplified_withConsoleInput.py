# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:21:57 2024

@author: Maciek
"""

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load data
df = pd.read_csv('breast_cancer.csv')


#Inputs to train
X_train = df.iloc[0:500, 2:7]
#Output of model to train
y_train = df.iloc[0:500, 1]

#Inputs to test
X_test = df.iloc[501:550, 2:7]
#Inputs to test
y_test = df.iloc[501:550, 1] 

# Initialize and train the perceptron model
perceptron = Perceptron(max_iter=10000, eta0=0.1, random_state=42)
perceptron.fit(X_train, y_train)
joblib.dump(perceptron, 'perceptron_model.pkl')

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

input_y_new_pred = perceptron.predict(new_prediction)
print(input_y_new_pred)
if(input_y_new_pred == 'M'):
    print('Cancer may be maligant :(')
else:
    print('Cancer should not be maligant :)')
print('done')
input('Press any key to close')