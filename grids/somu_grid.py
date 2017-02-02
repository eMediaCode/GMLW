
########### LOAD THE PARAM FILE ##############
print("[INFO] Loading the params file")
import json
with open("params.json") as datafile:
     params = json.load(datafile)


print("[INFO] params loaded")

if parmas["which_grid"] >0:
    print("[INFO] Grid Search params are loading ...")
    svm_grid = {
    "C":[0.001, 0.01, 1, 5 ,10, 100],
    "gamma": np.logspace(-9, 3, 13)
    }

    tree_grid = {
    "max_depth": np.arange(3, 10)
    }

    rf_grid = {
    "max_depth":[3,None],
    "max_features":sp_randint(1,11),
    "min_samples_split":sp_randint(1,11),
    "min_samples_leaf":sp_randint(1,11),
    "bootstrap":[True,False],
    "criterion":["gini","entropy"]
    }

    gbm_grid = {
    "n_estimators":[50,100,150,200],
    "max_depth":[3,None],
    "max_features":sp_randint(1,11),
    "max_leaf_nodes":sp_randint(3,11),
    "min_samples_leaf":sp_randint(1,11),
    "min_samples_split":sp_randint(1,11),
    "learning_rate":np.arange(0.1,0.6,0.1),
    "min_weight_fraction_leaf":np.arange(0,0.6,0.1)

    }

    xgb_grid = {
    "n_estimators": [100, 250, 500, 1000],
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
    "n_neighbours": [2 , 4, 8 , 16]
    }

else:
    print ("[INFO] Random search grid params are loading ...")
