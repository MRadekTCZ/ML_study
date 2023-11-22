# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:20:53 2023

@author: Maciek
"""

import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.data",
                   header = None, 
                   names = ['petal length', 'petal width', 
                            'sepal length', 'sepal width', 'species'])

xmodel = iris.iloc[:, :4]
ymodel = iris.loc[:, "species"]

categories = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}
ymodel=ymodel.apply(lambda x: categories[x])

X_attributes = xmodel.shape[1]
k_samples = xmodel.shape[0]

#Tablica srednich wartosci
x_av = []
sum_av = 0
for m in range(X_attributes):
    for k in range(k_samples):
        sum_av = sum_av + xmodel.iloc[k,m]    
    x_av.append(sum_av/k_samples)    
    sum_av = 0
#Tablica wspolczynnikow
a_coeff =[]
#Tablica wspolczynnikow wolnych
b_coeff = []


sum_nom = 0
sum_denom = 0

     