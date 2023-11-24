# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:20:53 2023

@author: Maciek
"""
#implementation of machine learning from scratch
#no ML libraries - only plot libs
import pandas as pd
import matplotlib.pyplot as plt
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

#Last row is output value (spiecie name)
y_model = iris.loc[:, "species"]

#Converting specie name (string) to numeric value
categories = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}
y_model=y_model.apply(lambda x: categories[x])

#Linear regression function. Determining the coefficients of the polynomial for each attribute 
#This function has to be done on computer - a lot of data to compute
#This function returs coefficients for polynomial function that can predict future outputs. 
#After computing (after learning phase), return of this function (coefficients of polynomial function) 
#can be implemented on external MCU e.g. microcontroller STM32 - 
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
    #a table
    a_coeff = []
    #free ratio table
    b_coeff = []
    sum_xiyi = 0
    sum_xixi = 0
    sum_xi = 0
    sum_yi = 0
    n = k_samples
    wage = [0.33, 0.01, 0.33, 0.33]
    Linear_aprox_all = []  
    #Counting an for every attribute
    for m in range(X_attributes):
        for k in range(k_samples):
            #From the formula for the coefficients of a linear function after linear regression
            #ax + b = y(x)
            sum_xiyi =sum_xiyi + (xmodel.iloc[k,m]*ymodel[k])   
            sum_xixi =sum_xixi + (xmodel.iloc[k,m]*xmodel.iloc[k,m])   
            sum_xi = sum_xi + xmodel.iloc[k,m]
            sum_yi = sum_yi + ymodel[k]
        
        #nominator    
        asum_nom = n * sum_xiyi - sum_xi*sum_yi
        #denominator
        asum_denom = n * sum_xixi - sum_xi*sum_xi
        an = asum_nom/asum_denom
        
        #nominator  
        bsum_nom = sum_yi - an * sum_xi
        #denominator
        bsum_denom = n
        bn = bsum_nom/bsum_denom      
        a_coeff.append(an)  
        b_coeff.append(bn) 
        sum_xiyi = 0
        sum_xixi = 0
        sum_xi = 0
        sum_yi = 0
        
    #All the coefficients are already determined at this point, 
    #but without further steps would have to have the same weights

    #Determination of linear functions from coefficients
    for m in range(len(a_coeff)):
        Linear_aprox = []
        xn = []
        y=xmodel.iloc[:,m].min() 
        dt = (xmodel.iloc[:,m].max()  - y)/k_samples     
        for k in range(k_samples):     
            y = y+dt
            xn.append(y)
            Linear_aprox.append((a_coeff[m]*y+b_coeff[m]))
        
        Linear_aprox_all.append(Linear_aprox)
        
    #Square Error calculation 
    Error_delta = []
    for m in range(X_attributes): 
        err = 0
        for k in range(k_samples):
              err = err +  (Linear_aprox_all[m][k] - ymodel[k])*(Linear_aprox_all[m][k] - ymodel[k])
        Error_delta.append(err/k_samples)   
    print(Error_delta)
    
    #Based on the squared error, 
    #it is possible to determine weights for individual attributes (their influence on the output value).
    #If the regression error is large, then the less influence an attribute should have on the prediction result.
    one_by_error = [1/e for e in Error_delta]
    
    wage = [(e/sum(one_by_error)) for e in one_by_error]
    print(wage)
    
    #Adding wages to coefficients    
    for m in range(len(a_coeff)):
        a_coeff[m] = a_coeff[m]*wage[m]
        b_coeff[m] = b_coeff[m]*wage[m]
    coeff = [a_coeff,b_coeff,wage]
    
    return coeff

#Predicting output on the basis of polynomial function coefficients
def LinearPrediction(coeff, Xinput):  
    Youtput = 0
    a_coeff = coeff[0]
    b_coeff = coeff[1]   
    m_coef = []
    for xn in range(len(a_coeff)):
        m_coef.append (Xinput[xn]*a_coeff[xn])
    epsilon = sum(b_coeff)
    Youtput = sum(m_coef) + epsilon
    return Youtput                   

coeffs = LinearLearning(x_model,y_model)

a_coeff = coeffs[0]
b_coeff = coeffs[1]

#Testing predictions
znany_kwiatek1 = [5, 3.2, 1.2, 0.2] #Iris Setosa
znany_kwiatek2 = [5.6, 3, 4.1, 1.3] #Iris Versicolor
znany_kwiatek3 = [6.2, 3.4, 5.4, 2.3] #Iris Virginica

#skaling input for 0:1 values
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

znany_kwiatek1 = scale_row_with_matrix(znany_kwiatek1,X)
znany_kwiatek2 = scale_row_with_matrix(znany_kwiatek2,X)
znany_kwiatek3 = scale_row_with_matrix(znany_kwiatek3,X)

Xinput = znany_kwiatek1                

print(LinearPrediction(coeffs,(znany_kwiatek1)))
print(LinearPrediction(coeffs,(znany_kwiatek2)))
print(LinearPrediction(coeffs,(znany_kwiatek3)))


#Printing plots and debugging
Classification = []
Numeric_Calssification = []
for row in range(x_model.shape[0]):    
    Youyput = LinearPrediction(coeffs,x_model.iloc[row,:])
    Numeric_Calssification.append(Youyput)
    if(Youyput < 1.4):
        Classification.append(categories["Iris-setosa"])
    elif(Youyput >= 1.4 and Youyput < 2.55 ):
        Classification.append(categories["Iris-versicolor"])
    elif(Youyput >= 2.55 ):
        Classification.append(categories["Iris-virginica"])

Linear_aprox_all = []   
wage =   coeffs[2] 
for m in range(len(a_coeff)):
    Linear_aprox = []
    xn = []
    y=x_model.iloc[:,m].min() 
    dt = (x_model.iloc[:,m].max()  - y)/150
    for k in range(150):     
        y = y+dt
        xn.append(y)
        Linear_aprox.append((a_coeff[m]*y+b_coeff[m])*(1/wage[m]))
    Linear_aprox_all.append(Linear_aprox) 
    plt.figure(figsize=(8, 6)) 
    plt.scatter(x_model.iloc[:, m], y_model, label='Punkty danych', color='blue')
    
    plt.plot(xn, Linear_aprox_all[m], label='Regresja liniowa', color='orange')

    plt.xlabel(f'Cecha {m+1}')  
    plt.ylabel('Wartość (5. kolumna)')  
    plt.title(f'Wykres cechy {m+1} względem wartości') 
    plt.grid(True) 
    plt.legend()  
    plt.show() 
        
#algorithm evaluation
evaluation = 0
for i in range(len(Classification)):
    
    if (Classification[i] ==  y_model[i]):
        evaluation = evaluation + 1
        
#While testing, my prediction is right in about 91%
print(evaluation / len(Classification)*100)
        



