# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:18:30 2023

@author: Maciek
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
def Normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

# Scalling
def Min_max_scale_dataframe(df):
    scaled_df = pd.DataFrame()
    for col in df.columns:
        scaled_df[col] = Normalize_column(df[col])
    return scaled_df

#Reading data
iris = pd.read_csv("iris.data",
                   header = None, 
                   names = ['petal length', 'petal width', 
                            'sepal length', 'sepal width', 'species'])
#First 4 columns are 4 attributes
X = iris.iloc[:, :4]
x_model = Min_max_scale_dataframe(X)

print(x_model.shape[0])
print(x_model.shape[1])
#Last row is output value (spiecie name)
y_model = iris.loc[:, "species"]
#Converting specie name (string) to numeric value
categories = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}
y_model=y_model.apply(lambda x: categories[x])
#where q is distance from one point in n-dimensional space
def Euclidian_dist(p, q,):
    multi_dim_dist = 0
    for n in range(len(p)):
        multi_dim_dist = ((q[n]-p[n])*(q[n]-p[n])) + multi_dim_dist
    multi_dim_dist = math.sqrt(multi_dim_dist)
    return multi_dim_dist                       
    
n1 = [1, 1, 1, 1]
n2 = [1, 1.2, 1.7, 1]

print(Euclidian_dist(n1, n2))

znany_kwiatek1 = [5, 3.2, 1.2, 0.2] #Iris Setosa
znany_kwiatek2 = [5.6, 3, 4.1, 1.3] #Iris Versicolor
znany_kwiatek3 = [6.2, 3.4, 5.4, 2.3] #Iris Virginica
other = [1, 2, 3, 4]
flowers = [znany_kwiatek1, znany_kwiatek2, znany_kwiatek3, other]
def scale_row_with_matrix(row, matrix):
    scaled_row = []
    for i in range(len(row)):
        column_values = matrix.iloc[:, i]
        min_val = min(column_values)
        max_val = max(column_values)
        scale = max_val - min_val
        scaled_value = (row[i] - min_val) / scale if scale != 0 else row[i]
        scaled_row.append(scaled_value)
    return scaled_row
#inputs should be scalled to make equal wage for different factors 
znany_kwiatek1 = scale_row_with_matrix(znany_kwiatek1,X)
znany_kwiatek2 = scale_row_with_matrix(znany_kwiatek2,X)
znany_kwiatek3 = scale_row_with_matrix(znany_kwiatek3,X)

#K nearest neighbors algorithm - i used it for k = 1
def KNN(New_data, Base_data):
    distances = []
    for n in range(Base_data.shape[0]):
        d = Euclidian_dist(New_data, Base_data.iloc[n,:])
        distances.append(d)
    return distances

def KNN_det(distances,Y_model, k):
    shortest_distance = min(distances)
    i_short = distances.index(shortest_distance)
    return y_model[i_short]



Classification = []

evaluation = 0
for row in range(len(y_model)):
     KNN_distances = []
     KNN_distances = KNN(x_model.iloc[row,:], x_model)
     Y_det = KNN_det(KNN_distances, y_model)
     Classification.append(Y_det)

for i in range(len(Classification)):
    
    if (Classification[i] ==  y_model[i]):
        evaluation = evaluation + 1
        
#While testing, my prediction is right in about 91%
print(evaluation / len(Classification)*100)        


KNN_distances = []
KNN_distances = KNN(other, x_model)
Y_det = KNN_det(KNN_distances, y_model)

print(Y_det)