# We start by loading libraries. The most popular ones are
# pandas - for working with data
# matplotlib - for drawing graphs
# sklearn - containing ready-made functions for modeling data
 
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
 
# here we load the data into a data frame object from the pandas library
# the CSV file has no header, so header=None
# We name the columns using the names parameter
# In ML scripts, you have to get the data from somewhere, so knowing the command 
# read_csv is super useful
 
iris = pd.read_csv("iris.data",
                   header = None, 
                   names = ['petal length', 'petal width', 
                            'sepal length', 'sepal width', 'species'])
iris.head()
 
from sklearn.linear_model import LinearRegression
xmodel = iris.iloc[:, :4]
ymodel = iris.loc[:, "species"]

categories = {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}
ymodel=ymodel.apply(lambda x: categories[x])

xmodel.head()
ymodel.head()

lr = LinearRegression()
lr.fit(xmodel,ymodel)
lr.score(xmodel,ymodel)

iris_1 = [5, 3.5, 1.4, 0.2]
iris_2 = [6.4, 3, 4.5, 1]
iris_3 = [6, 3, 5, 2]
other = [1, 2, 3, 4]
flowers = [iris_1, iris_2, iris_3, other]

species_predict = lr.predict(flowers)
print(species_predict)

# you can check the size of the loaded set
# if the object has more dimensions, you can independently check each of them
# In ML scripts, you often need to initialize the sizes of other objects depending on the
# the size of the input data. This is done by using the shape property
iris.shape
iris.shape[0]
iris.shape[1]
i =8
# further prepare the chart - here determining the min and max values for the 
# 2 selected columns with flower sizes. When you want to refer to an entire column in the data frame,
# then in square brackets you give the name of this column
x_min, x_max = iris['petal length'].min() - .5, iris['petal length'].max() + .5
y_min, y_max = iris['petal width'].min() - .5, iris['petal width'].max() + .5
 
# each species to be displayed in a different color - we define the dictionary
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}
 
# create an object responsible for the chart to be drawn and its coordinates
# instructions from now until plt.show() run by selecting this entire block of code
fig, ax = plt.subplots(figsize=(8, 6))
 
# we group the data by species and draw the data. Here we use the groupby method of the data frame object
# function returns a key identifying the name of the group (here it is the name of the flower species) and the
# samples included in this group. This allows us to draw each group in a different color
for key, group in iris.groupby(by='species'):
    plt.scatter(group['petal length'], group['petal width'], 
                c=colors[key], label=key)
 
# add legend and axis description
ax.legend()
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title("IRIS DATASET CATEGORIZED")
 
plt.show()
 
# now a similar graph can be made for sepal
# remember to run having the code block selected from now until plt.show()
# the steps are the same as in the previous example
x_min, x_max = iris['sepal length'].min() - .5, iris['sepal length'].max() + .5
y_min, y_max = iris['sepal width'].min() - .5, iris['sepal width'].max() + .5
 
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}
 
fig, ax = plt.subplots(figsize=(8, 6))
 
for key, group in iris.groupby(by='species'):
    # scatter function takes arguments - X coordinates of points, Y coordinates of points,
    # color and name of the group being drawn
    plt.scatter(group['sepal length'], group['sepal width'], 
                c=colors[key], label=key)
 
ax.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title("IRIS DATASET CATEGORIZED")
 
plt.show()

# create a chart consisting of 4 small charts
fig, ax = plt.subplots(2,2,figsize=(10, 6))
 

plt_position = 1
 
# we illustrate the relationship between this variable and the other characteristics of the samples
feature_x= 'petal width'
 
for feature_y in iris.columns[:4]:
 
    plt.subplot(2, 2, plt_position) 
 
    for species, color in colors.items():
        # when drawing, filter out only flowers of one species
        # see how to filter the data. This is done by the loc function called for the data frame
        #The expression in square brackets is to return True/False. The rows returned will be,
        # where the expression has the value True. After the comma is the name of the column to be returned
        plt.scatter(iris.loc[iris['species']==species, feature_x],
                    iris.loc[iris['species']==species, feature_y],
                    label=species,
                    alpha = 0.45, # transparency
                    color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt_position += 1
 
plt.show()

print("Koniec")
