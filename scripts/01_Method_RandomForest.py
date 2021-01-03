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
from sklearn.model_selection import cross_val_score
from datetime import datetime
import pickle as pkl

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
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [50, 100, 200, 500]
# Minimum number of samples required at each leaf node
min_samples_leaf = [20, 40, 100]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#%% Run base model
RFBase = RandomForestRegressor(n_estimators = 1000, max_features = "sqrt", bootstrap = True, random_state = 2312, verbose = 3)

# Cross validation

RFModel = RFBase.fit(X_tr, y_tr)
cross_val_score(RFBase, X_tr, y_tr, scoring = ("neg_mean_squared_error")) 



# Prediction and check error
prediction = RFModel.predict(X_t)
RSME = mean_squared_error(y_t, prediction).round(2)
print(RSME)

#%% Excecuting grid search
# Base model for tuning
RFBase = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
RFGridModel = RandomizedSearchCV(estimator = RFBase, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

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

#%%
baseLine = {"baseLine" : np.repeat(np.mean(y_t), 46545)}
baseLine = pd.DataFrame(data = baseLine  )

mean_squared_error(y_t, baseLine).round(2)

#%%
baseLine = {"baseLine" : np.repeat(41, 46545)}
baseLine = pd.DataFrame(data = baseLine)

mean_squared_error(y_t, baseLine).round(2)
