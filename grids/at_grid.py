import math

########### LOAD THE PARAM FILE ##############
print("[INFO] Loading the params file")
import json
with open("params.json") as datafile:
     params = json.load(datafile)


print("[INFO] params loaded")

if parmas["which_grid"] >0:
    print("[INFO] Grid Search params are loading ...")

    tree_grid = {
    "max_depth": np.arange(3, 10)
    }

    rf_grid = {
    "n_estimators":[120, 300, 500, 800, 1200],
    "max_depth":[5, 8, 15, 25, 30, None],
    "max_features":[math.log(2), math.sqrt(2), None],
    "min_samples_split":[1, 2, 5, 10, 15, 100],
    "min_samples_leaf":[1, 2, 5, 10],
    }

    xgb_grid = {
    "eta": [0.01,0.015,0.025,0.05,0.1],
    "max_depth": [3, 5 , 7, 9 , 12 , 15 ,17 , 25],
    "gamma":[0.05,0.1,0.3,0.5,0.7,0.9,1.0],
    "min_child_weight":[1, 3, 5, 7],
    "lambda":[0.01, 0.1, 1.0],
    "alpha":[0, 0.1, 0.5, 1.0],
    "subsample": [0.6,0.7,0.8,0.9,1.0],
    "colsample_bytree": [0.6,0.7,0.8,0.9,1.0]
    }

    lasso_grid = {
    "alpha": [0.1, 1.0, 10],
    "normalize": [True, False]
    }

    ridge_grid = {
    "alpha": [0.01, 0.1, 1.0, 10, 100],
    "fit_intercept": [ True, False],
    "normalize": [ True, False],
    }

    knn_grid = {
    "n_neighbours": [2 , 4, 8 , 16],
    "p": [2, 3]
    }

    svm_grid= [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
