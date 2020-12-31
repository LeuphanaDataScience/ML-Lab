# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:24:27 2020

@author: Shaurya
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import pickle as pkl

#%%
# Control panel
scoreNormalized = False
featureNormalized = True

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

# # correcting a mistake - encoding the time signature variable as dummy
# X_tr.info()
# X_tr.time_signature.replace((1,0.75,1.25,0.25,0), ('4/4','3/4','5/4','1/4','0/4'), inplace=True)
# X_tr["time_signature"].value_counts()
# dum_df = pd.get_dummies(X_tr["time_signature"], columns=["time_signature"], prefix="time_sig_is")
# X_tr = X_tr.join(dum_df)
# X_tr = X_tr.drop(["time_signature"],1)

# X_t.time_signature.replace((1,0.75,1.25,0.25,0), ('4/4','3/4','5/4','1/4','0/4'), inplace=True)
# X_t['time_signature'] = X_t['time_signature'].astype(object)
# dum_df = pd.get_dummies(X_t["time_signature"], columns=["time_signature"], prefix="time_sig_is")
# X_t = X_t.join(dum_df)
# X_t = X_t.drop(["time_signature"],1)

# # removing the extra dummy to avoid the dummy variable trap
# X_tr = X_tr.drop(columns=['time_sig_is_0/4'])
# X_t = X_t.drop(columns=['time_sig_is_0/4'])

# # correcting another mistake - removing an extra dummy variable in Genre
# X_tr = X_tr.drop(columns=['genre_is_A Capella'])
# X_t = X_t.drop(columns=['genre_is_A Capella'])

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
    # Calculate mean and std for the training set
    scaler = StandardScaler().fit(X_tr.loc[:,"acousticness":"valence"])
    
    # Normalize training set
    X_tr.loc[:,"acousticness":"valence"] = scaler.transform(X_tr.loc[:,"acousticness":"valence"])
    
    # Normalize test set based on training set param eter
    X_t.loc[:,"acousticness":"valence"] = scaler.transform(X_t.loc[:,"acousticness":"valence"])
    
# Convert column into 1D array (for training labels)
y_tr = np.array(y_tr).ravel()


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

