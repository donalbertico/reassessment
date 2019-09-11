import pandas as pd
import numpy as np

cars = pd.read_csv('../data/imports-85.data',sep=',', header = None)

cars.columns = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

# print(cars.describe())

cars.replace("?", np.NaN)
missing_cols = []

for col in cars.columns:
    print('?' in cars[col])
    if '?' in cars[col]:
        missing_cols.append(col)

print(cars)
