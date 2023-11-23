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

x_model = iris.iloc[:, :4]
y_model = iris.loc[:, "species"]

categories = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}
y_model=y_model.apply(lambda x: categories[x])


def LinearLearning(xmodel, ymodel):
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
    a_coeff = []
    #Tablica wspolczynnikow wolnych
    b_coeff = []
    sum_xiyi = 0
    sum_xixi = 0
    sum_xi = 0
    sum_yi = 0
    n = k_samples
    wage = [0.33, 0.01, 0.33, 0.33]
    #Couting an for every attribute
    for m in range(X_attributes):
        for k in range(k_samples):
            sum_xiyi =sum_xiyi + (xmodel.iloc[k,m]*ymodel[k])   
            sum_xixi =sum_xixi + (xmodel.iloc[k,m]*xmodel.iloc[k,m])   
            sum_xi = sum_xi + xmodel.iloc[k,m]
            sum_yi = sum_yi + ymodel[k]
        asum_nom = n * sum_xiyi - sum_xi*sum_yi
        asum_denom = n * sum_xixi - sum_xi*sum_xi
        an = asum_nom/asum_denom
        bsum_nom = sum_yi - an * sum_xi
        bsum_denom = n
        bn = bsum_nom/bsum_denom      
        a_coeff.append(an*wage[m])  
        b_coeff.append(bn*wage[m]) 
        sum_xiyi = 0
        sum_xixi = 0
        sum_xi = 0
        sum_yi = 0
    coeff = [a_coeff,b_coeff]
    return coeff

def LinearPrediction(coeff, Xinput):  
    Youtput = 0
    a_coeff = coeff[0]
    b_coeff = coeff[1]   
    m_coef = []
    for xn in range(len(a_coeff)):
        m_coef.append (Xinput[xn]*a_coeff[xn])
    epsilon = sum(b_coeff)
    Youtput = sum(m_coef + epsilon)
    return Youtput                   

coeffs = LinearLearning(x_model,y_model)

a_coeff = coeffs[0]
b_coeff = coeffs[1]


znany_kwiatek1 = [5, 3.2, 1.2, 0.2] #Iris Setosa
znany_kwiatek2 = [5.6, 3, 4.1, 1.3] #Iris Versicolor
znany_kwiatek3 = [6.2, 3.4, 5.4, 2.3] #Iris Virginica




Xinput = znany_kwiatek1                

print(LinearPrediction(coeffs,znany_kwiatek1))
print(LinearPrediction(coeffs,znany_kwiatek2))
print(LinearPrediction(coeffs,znany_kwiatek3))

Classification = []


for row in range(x_model.shape[0]):    
    Youyput = LinearPrediction(coeffs,x_model.iloc[row,:])
    if(Youyput < 1.25):
        Classification.append(categories["Iris-setosa"])
    elif(Youyput >= 1.25 and Youyput < 2.75 ):
        Classification.append(categories["Iris-versicolor"])
    elif(Youyput >= 2.75 ):
        Classification.append(categories["Iris-virginica"])

Linear_aprox_all = []   
wage = [0.33, 0.01, 0.33, 0.33]     
for m in range(len(a_coeff)):
    Linear_aprox = []
    xn = []
    if m==0:
        y=4.0
        dt = (8.0 - y)/150
    elif m==1:
        y=2.0
        dt = (4.5 - y)/150
    elif m==2:
        y=1.0
        dt = (7.0 - y)/150
    elif m==3:
        y=0.0
        dt = (2.5 - y)/150        
    for k in range(150):     
        y = y+dt
        xn.append(y)
        Linear_aprox.append((a_coeff[m]*y+b_coeff[m])*(1/wage[m]))
    Linear_aprox_all.append(Linear_aprox) 
    plt.figure(figsize=(8, 6))  # Rozmiar wykresu
    plt.scatter(x_model.iloc[:, m], y_model, label='Punkty danych', color='blue')
    # Tworzenie wykresu regresji liniowej
    plt.plot(xn, Linear_aprox_all[m], label='Regresja liniowa', color='orange')

    plt.xlabel(f'Cecha {m+1}')  # Etykieta osi X
    plt.ylabel('Wartość (5. kolumna)')  # Etykieta osi Y
    plt.title(f'Wykres cechy {m+1} względem wartości')  # Tytuł wykresu
    plt.grid(True)  # Włączenie siatki na wykresie
    plt.legend()  # Dodanie legendy
    plt.show()  # Wyświetlenie wykres
        



