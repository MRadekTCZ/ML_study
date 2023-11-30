# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:18:51 2023

@author: Maciek
"""


 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

 
class Perceptron:
    
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    
    def predict(self, x):
        
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        #activation = self.get_activation(x_1)
        #y_pred = np.where(activation >0, 1, -1)
        #return y_pred
        return np.where(self.get_activation(x_1) > 0, 1, -1)
        
    
    def get_activation(self, x):
        
        activation = np.dot(x, self.w)
        return activation
     
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)
 
        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):
 
            error = 0
            
            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y - activation), X_1)
            self.w += delta_w
                
            error = np.square(y - activation).sum()/2.0
                
            self.list_of_errors.append(error)
            
            if(self.is_verbose):
                print("Epoch: {}, weights: {}, error {}".format(
                        e, self.w, error))
                
diag = pd.read_csv("breast_cancer.csv",
                   header = None)

xmodel = diag.iloc[1:500, 2:32]
xtest = diag.iloc[501:, 2:32]
ymodel = diag.iloc[1:500, 1]
ytest = diag.iloc[501:, 1]
categories = {"M":1, "B":-1}
ymodel=ymodel.apply(lambda x: categories[x])
ytest=ytest.apply(lambda x: categories[x])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(xmodel)
X_std = scaler.transform(xmodel)
 
perceptron = Perceptron(eta=0.00001, epochs=1000)
perceptron.fit(X_std, ymodel)
plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)


xtest = scaler.transform(xtest)
y_pred = perceptron.predict(xtest)
 
good = ytest[ytest == y_pred].count()
total = ytest.count()
print('result: {}'.format(100*good/total))