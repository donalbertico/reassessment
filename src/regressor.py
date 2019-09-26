import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import train_test_split

cars = pd.read_csv("../data/cars.csv")
wines = pd.read_csv("../data/wines.csv")
concrete = pd.read_csv("../data/concrete.csv")
cars.columns = ['mpg','displacement','horsepower','weight','accelerator']
wines.columns = ['quality','fixed acidity','density','sulphates','alcohol']

cv = ShuffleSplit(10,test_size=0.2, random_state =42)


lmodel = linear_model.SGDRegressor(penalty = "l1")
dTmodel = DecisionTreeRegressor(max_depth = 3,random_state= 23)


independent =  ['displacement','horsepower','weight','accelerator']
cars_x_train, cars_x_test, cars_y_train, cars_y_test = train_test_split(cars[independent],cars['mpg'],test_size=0.2)

x = cars_x_train
y = cars_y_train

lmodel.fit(x,y)
dTmodel.fit(x,y)

print('cars intercept: ', lmodel.intercept_)
print('cars coeficients ',lmodel.coef_)
print('LR cross val score', cross_val_score(lmodel,x,y,cv=cv).mean() )
print('DT cross val score', cross_val_score(dTmodel,x,y,cv=cv).mean() )

independent =  ['fixed acidity','density','sulphates','alcohol']
wines_x_train, wines_x_test, wines_y_train, wines_y_test = train_test_split(wines[independent],wines['quality'],test_size=0.2)

x = wines_x_train
y = wines_y_train

lmodel.fit(x,y)
dTmodel.fit(x,y)

print('wines intercept', lmodel.intercept_)
print('wines coefitient',lmodel.coef_)
print('LR cross val score', cross_val_score(lmodel,x,y,cv=cv).mean() )
print('DT cross val score', cross_val_score(dTmodel,x,y,cv=cv).mean() )

independent =  ['cement','water','age']
concrete_x_train, concrete_x_test, concrete_y_train, concrete_y_test = train_test_split(concrete[independent],concrete['strength'],test_size=0.2)

x = concrete_x_train
y = concrete_y_train

lmodel.fit(x,y)
dTmodel.fit(x,y)

print('concrete intercept', lmodel.intercept_)
print('concrete coeficient',lmodel.coef_)
print('LR cross val score', cross_val_score(lmodel,x,y,cv=cv).mean() )
print('DT cross val score', cross_val_score(dTmodel,x,y,cv=cv).mean() )
