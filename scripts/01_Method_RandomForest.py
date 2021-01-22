# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:00:35 2020

This file implements Random Forest on the Spotify Popularity Score Prediction

@author: Chan Le
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from datetime import datetime
import pickle as pkl
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import catboost as cb

#%%
# Control panel
scoreNormalized = False
featureNormalized = True

#%% Preprocessing
# Load data
X_t = pd.read_csv("data/X_t.csv")
X_tr = pd.read_csv("data/X_tr.csv")
y_t = pd.read_csv("data/y_t.csv")
y_tr = pd.read_csv("data/y_tr.csv")


# Normalize popularity score
if scoreNormalized == True:
    
    # Calculate mean and std for the training set
    trainStd = np.std(y_tr["popularity"])
    trainMean = np.mean(y_tr["popularity"])
    
    # Normalize training set and test set based on training set parameter
    y_tr["popularity"] = (y_tr["popularity"]  - trainMean) / trainStd
    y_t["popularity"] = (y_t["popularity"]  - trainMean) / trainStd
 
# Normalize features   
if featureNormalized == True: 
    #needNormalize = ["duration_ms", "energy", "instrumentalness", "key", "loudness", "tempo"]
    
    # Calculate mean and std for the training set
    scaler = StandardScaler().fit(X_tr.loc[:,"acousticness":"valence"])
    
    # Normalize training set
    X_tr.loc[:,"acousticness":"valence"] = scaler.transform(X_tr.loc[:,"acousticness":"valence"])
    
    # Normalize test set based on training set parameter
    X_t.loc[:,"acousticness":"valence"] = scaler.transform(X_t.loc[:,"acousticness":"valence"])
    
# Convert column into 1D array (for training labels)
y_tr = np.array(y_tr).ravel()


#%% Grid search
# Prepare grid hyperparameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [40, 60, 80]
# Minimum number of samples required to split a node
min_samples_split = [25, 50, 75]
# Minimum number of samples required at each leaf node
min_samples_leaf = [10, 20, 30]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Decrease of the impurity for a split
min_impurity_decrease = [float(x) for x in [0, 0.1, 0.01]]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'min_impurity_decrease': min_impurity_decrease}

#%% Run base model
RFBase = RandomForestRegressor(n_estimators = 1000, max_features = "sqrt", bootstrap = True, random_state = 2312, verbose = 3, max_samples = 0.3)

# Cross validation
RFCV = cross_val_score(RFBase, X_tr, y_tr, scoring=('neg_mean_squared_error'))
RFR2 = cross_val_score(RFBase, X_tr, y_tr, scoring=('r2'))

np.mean(RFR2)
np.std(RFR2)


RFBase.fit(X_tr, y_tr)
#%% Excecuting grid search
# Base model for tuning
RFBase = RandomForestRegressor(n_estimators = 1000, max_features = "sqrt", bootstrap = True, max_samples = 0.3, random_state = 2312, min_impurity_decrease= None)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
RFGridModel = GridSearchCV(estimator = RFBase, param_grid = random_grid, 
                           scoring = "neg_mean_squared_error", 
                           cv = 3, verbose = 5, n_jobs = 8)

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


#%% Base line model error calculation
baseLine = {"baseLine" : np.repeat(np.mean(y_t), 46545)}
baseLine = pd.DataFrame(data = baseLine  )

mean_squared_error(y_t, baseLine).round(2)

#%% Grid Search for XGBoost

learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
max_iter = [1000, 2000, 2500]
l2_reg = [0.01, 0.05, 0.07, 0.1, 0.15]
max_bins = [100, 200]

histBoost_grid = {'learning_rate': learning_rate,
                  'max_iter' : max_iter,
                  'l2_regularization': l2_reg,
                  'max_bins': max_bins}

HistBoostBase = HistGradientBoostingRegressor(max_bins= 100, random_state = 2312)

HistBoostGrid = GridSearchCV(estimator = HistBoostBase, param_grid=histBoost_grid,
                             scoring = 'neg_mean_squared_error', 
                             cv = 3, verbose = 5, n_jobs = 4)

HistBoostGrid.fit(X_tr, y_tr)

#%% Grid Search for XGBoost

params = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [6, 8, 10],
         'iterations': [300, 500]}

CatBoostBase = cb.CatBoostRegressor()

CatBoostGrid = GridSearchCV(estimator = CatBoostBase, param_grid=params,
                             scoring = 'neg_mean_squared_error', 
                             cv = 3, verbose = 5, n_jobs = 8)

CatBoostGrid.fit(X_tr, y_tr)

#%% Post-model analysis
# Check for best params
CatBoostGrid.best_params_

# Prediction and check error
prediction = CatBoostGrid.predict(X_t)
RSME = mean_squared_error(y_t, prediction).round(2)
print(RSME)
#%% Save model
# Get timestamp
now = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")

# Write to pickle
with open(f"models/CatBoost_{now}_BestRMSE_{RSME}.pickle", "wb") as f:
    pkl.dump(CatBoostGrid, f)  