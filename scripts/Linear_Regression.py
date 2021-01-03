# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:24:27 2020

@author: Shaurya
"""

import numpy as np
import pandas as pd
from datetime import datetime
import pickle as pkl
from sklearn import preprocessing

#%% Preprocessing
# Load data
X_t = pd.read_csv("C:/Users/Shaurya/Documents/GitHub/ML-Lab/data/X_t.csv")
X_tr = pd.read_csv("C:/Users/Shaurya/Documents/GitHub/ML-Lab/data/X_tr.csv")
y_t = pd.read_csv("C:/Users/Shaurya/Documents/GitHub/ML-Lab/data/y_t.csv")
y_tr = pd.read_csv("C:/Users/Shaurya/Documents/GitHub/ML-Lab/data/y_tr.csv")

# Rename ID column
for data in [X_t, X_tr, y_t, y_tr]:
    data = data.rename(columns= {"Unnamed: 0":"ID"}, inplace= True)

# Set ID column aside
ID_t = X_t["ID"]
ID_tr = X_tr["ID"]

# Drop ID column 
X_t = X_t.drop("ID", axis = 1)
X_tr = X_tr.drop("ID", axis = 1)
y_t = y_t.drop("ID", axis = 1)
y_tr = y_tr.drop("ID", axis = 1)

# scaling the inputs - do it only for the continuous variables
X_tr_scaled = pd.DataFrame(preprocessing.scale(X_tr[['acousticness','danceability',
                                                     'duration_ms','energy',
                                                     'instrumentalness','liveness',
                                                     'loudness','speechiness','tempo',
                                                     'valence']]))

# keeping the column names
X_tr_scaled.columns = ['acousticness','danceability','duration_ms','energy',
                       'instrumentalness','liveness','loudness','speechiness',
                       'tempo','valence']

# adding the other columns (in their original form) to the scaled data
X_tr_scaled = pd.concat([X_tr_scaled, X_tr.iloc[:, 12:42]], axis=1)

# adding the 2 remaining columns
X_tr_scaled = pd.concat([X_tr_scaled, X_tr[['key', 'Mode is Major']]], axis=1)

# importing models and evaluation metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# running linear regression without regularisation
linreg = LinearRegression()

# cross validation scores; 5 folds by default
print(np.mean(cross_val_score(linreg, X_tr, y_tr, scoring=('neg_mean_squared_error'))), 
      np.std(cross_val_score(linreg, X_tr, y_tr, scoring=('neg_mean_squared_error')))) 

print(np.mean(cross_val_score(linreg, X_tr, y_tr, scoring=('r2'))), 
      np.std(cross_val_score(linreg, X_tr, y_tr, scoring=('r2')))) 

############# Maybe this can be done after the next milestone ####################################

# maybe calculate r square too

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
# https://scikit-learn.org/stable/modules/grid_search.html

# regul;arised models will always perform worse as they are trying to prevent overfitting

# running linear regression with l2 regularisation - ridge
ridge = Ridge(alpha=0.1)
print(np.mean(cross_val_score(ridge, X_tr, y_tr, scoring=('neg_mean_squared_error'))), 
      np.std(cross_val_score(ridge, X_tr, y_tr, scoring=('neg_mean_squared_error')))) 
print(np.mean(cross_val_score(ridge, X_tr, y_tr, scoring=('r2'))), 
      np.std(cross_val_score(ridge, X_tr, y_tr, scoring=('r2'))))

# running linear regression with l1 regularisation - lasso
lasso = Lasso(alpha=0.1)
print(np.mean(cross_val_score(lasso, X_tr, y_tr, scoring=('neg_mean_squared_error'))), 
      np.std(cross_val_score(lasso, X_tr, y_tr, scoring=('neg_mean_squared_error')))) 
print(np.mean(cross_val_score(lasso, X_tr, y_tr, scoring=('r2'))), 
      np.std(cross_val_score(lasso, X_tr, y_tr, scoring=('r2'))))

# running linear regression with l1 and l2 both - elasticnet
elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
print(np.mean(cross_val_score(lasso, X_tr, y_tr, scoring=('neg_mean_squared_error'))), 
      np.std(cross_val_score(lasso, X_tr, y_tr, scoring=('neg_mean_squared_error')))) 
print(np.mean(cross_val_score(lasso, X_tr, y_tr, scoring=('r2'))), 
      np.std(cross_val_score(lasso, X_tr, y_tr, scoring=('r2'))))
# recursive feature elimination

# NOTE: all models must be cross-validated to test performance and tune hyperparameters (random se)




#%% Grid search
# Prepare grid hyperparameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5, 10, 20, 50]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#%% Excecuting grid search
# Base model for tuning
RFBase = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
RFGridModel = RandomizedSearchCV(estimator = RFBase, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Execute grid search
RFGridModel.fit(X_tr, y_tr)

#%% Post-model analysis
# Check for best params
RFGridModel.best_params_

# Prediction and check error
prediction = RFGridModel.predict(X_t)
RSME = mean_squared_error(y_t, prediction).round(2)
print(RSME)

#%% Save model
# Get timestamp
now = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")

# Write to pickle
with open(f"models/RF_{now}_BestRMSE_{RSME}.pickle", "wb") as f:
    pkl.dump(RFGridModel, f)  

