#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv("car.csv")
df.head()

#counting null values in each cell
df.isnull().sum()

df.info()

df.columns

#removing car name column
car = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


#getting how old a car is as a column
from datetime import datetime
a = datetime.today().year
car['Current_Year'] = a

car['Years_old'] = car['Current_Year']-car['Year']

#dropping unnecessary columns
car.drop(['Year','Current_Year'],inplace = True, axis = 1)


#converting categorial features using oneHotEncoding
car = pd.get_dummies(car)


corr_mat = car.corr()


sns.heatmap(corr_mat, annot = True, cmap="RdYlGn")


X = car.iloc[:,1:]
y = car.iloc[:,0]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()


import numpy as np
from sklearn.model_selection import RandomizedSearchCV


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


rf_random.fit(X_train,y_train)
rf_random.best_params_


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)




