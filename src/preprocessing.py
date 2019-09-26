import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing


def dropOutlier(df,cond):
    drop = df[cond]
    df.drop(drop.index,inplace=True)

min_max_scaler = preprocessing.MinMaxScaler()

cars = pd.read_csv('../data/auto-mpg.data',header=None,delimiter='\s+')
wines = pd.read_csv('../data/winequality-red.csv',delimiter=';')
wines1 = pd.read_csv('../data/winequality-white.csv',delimiter=';')
concrete = pd.read_excel('../data/Concrete_Data.xls')

# wines = pd.concat([wines,wines1])

cars.columns = ['mpg','cylinders','displacement','horsepower','weight','accelerator','model','origin','car name']

cars = cars.replace("?", np.NaN)
cars['horsepower'] = pd.to_numeric(cars['horsepower'])

print(cars.dtypes)

plt.scatter(cars['mpg'],cars['accelerator'], color='red')
plt.ylabel('accelerator')
plt.xlabel('mpg')
plt.grid(True)
# plt.show()

plt.scatter(cars['mpg'],cars['displacement'], color='red')
plt.ylabel('displacement')
plt.xlabel('mpg')
plt.grid(True)
# plt.show()

plt.scatter(cars['mpg'],cars['horsepower'], color='red')
plt.ylabel('horsepower')
plt.xlabel('mpg')
plt.grid(True)
# plt.show()

plt.scatter(cars['mpg'],cars['weight'], color='red')
plt.ylabel('weight')
plt.xlabel('mpg')
plt.grid(True)
# plt.show()

plt.scatter(cars['mpg'],cars['cylinders'], color='red')
plt.ylabel('cylinders')
plt.xlabel('mpg')
plt.grid(True)
# plt.show()

independent =  ['mpg','displacement','horsepower','weight','accelerator']
cars = cars[independent]

cars.dropna(inplace=True)

dropOutlier(cars,(cars['displacement']>200) & (cars['mpg']>25))
dropOutlier(cars,(cars['horsepower']>125) & (cars['mpg']>25))
dropOutlier(cars,(cars['horsepower']>200) & (cars['mpg']<20))
dropOutlier(cars,(cars['horsepower']<75) & (cars['mpg']<20))
dropOutlier(cars,(cars['accelerator']>12) & (cars['mpg']>40))
dropOutlier(cars,(cars['accelerator']>10) & (cars['mpg']<15))

columns = cars.columns
cars = pd.DataFrame(min_max_scaler.fit_transform(cars.values))
cars.columns = cars.columns
cars.to_csv('../data/cars.csv', index=False)

wines = wines.replace("?", np.NaN)

print(wines.dtypes)

independent = (value for value in wines.columns if value != 'quality')
wines_independent = wines[independent]

plt.scatter(wines['quality'],wines['fixed acidity'], color='red')
plt.ylabel('fixed acidity')
plt.xlabel('quality')
plt.grid(True)
# plt.show()

plt.scatter(wines['quality'],wines['density'], color='red')
plt.ylabel('density')
plt.xlabel('quality')
plt.grid(True)
# plt.show()

plt.scatter(wines['quality'],wines['sulphates'], color='red')
plt.ylabel('sulphates')
plt.xlabel('quality')
plt.grid(True)
# plt.show()

plt.scatter(wines['quality'],wines['alcohol'], color='red')
plt.ylabel('alcohol')
plt.xlabel('quality')
plt.grid(True)
# plt.show()

wines = wines[['quality','fixed acidity','density','sulphates','alcohol']]

columns = wines.columns
wines = pd.DataFrame(min_max_scaler.fit_transform(wines.values))
wines.columns = wines.columns
wines.to_csv('../data/wines.csv', index=False)

concrete = concrete.replace("?", np.NaN)

concrete.columns = ['cement','blast furnace slag','fly ash','water','superplasticizer','coarse aggreagete','fine aggregate','age','strength']

plt.scatter(concrete['strength'],concrete['cement'], color='red')
plt.ylabel('cement')
plt.xlabel('strength')
plt.grid(True)
# plt.show()

plt.scatter(concrete['strength'],concrete['water'], color='red')
plt.ylabel('water')
plt.xlabel('strength')
plt.grid(True)
# plt.show()

plt.scatter(concrete['strength'],concrete['age'], color='red')
plt.ylabel('age')
plt.xlabel('strength')
plt.grid(True)
# plt.show()


concrete = concrete[['strength','cement','water','age']]
columns = concrete.columns
concrete = pd.DataFrame(min_max_scaler.fit_transform(concrete.values))
concrete.columns = columns
concrete.to_csv('../data/concrete.csv', index=False)
