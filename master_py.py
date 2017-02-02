""" MASTER PYTHON FILE """


import numpy as np
import pandas as pd
from models.base_line import logistic_cp, lasso_cp, ridge_cp, lasso_cp_cv , ridge_cp_cv
from models.knn import knn_cp, knn_cp_cv
from models.svm import svm_cp, svm_cp_cv
from models.tree import tree_cp, tree_cp_cv, rf_cp, rf_cp_cv, gbm_cp, gbm_cp_cv
from models.xgboost import xgb_cp, xgb_cp_cv
from grids import at_grid


######## CREATE A models_dump FOLDER IN THE PRESENT LOCATION #######
import os
if not os.path.exists("models_dump"):
    os.makedirs("models_dump")

########### LOAD THE PARAM FILE ##############
print("[INFO] Loading the params file")
import json
with open("params.json") as datafile:
     params = json.load(datafile)

print("[INFO] Loading Data")
X_dev = pd.read_csv(params["input_train_dataset"])

############# LOGISTIC REGRESSION ###########
if params["is_logistic"] > 0:
    print ("[INFO] Building logistic regression ...")
    logistic_cp(X_dev, Y_dev, params["model_dump_logistic"])


############# NAIVE BAYES ###########
if params["is_naive_bayes"] > 0:
    print("[INFO] building naive bayes ...")
    naive_bayes_cp(X_dev, Y_dev, params["model_dump_naive_bayes"])

############# LASSO REGRESSION ###########
if params["is_lasso"] >0:
    print ("[INFO] building Lasso Regression ...")
    if params["is_cv_req_lasso"] > 0:
        lasso_cp_cv(X_dev, Y_dev,lasso_grid, params["model_dump_lasso"])

    else:
        lasso_cp(X_dev, Y_dev, params["model_dump_lasso"])

############# RIDGE REGRESSION ###########
if params["is_ridge"] >0:
    print ("[INFO] building ridge regression ...")
    if params["is_cv_req_ridge"] > 0:
        ridge_cp_cv(X_dev, Y_dev,ridge_grid, params["model_dump_ridge"])
    else:
        ridge_cp(X_dev, Y_dev, params["model_dump_ridge"])

############# KNN ###########
if params["is_knn"] >0:
    print("[INFO] building KNN's ...")
    if params["is_cv_req_knn"]>0:
        knn_cp_cv(X_dev, Y_dev, knn_grid, params["model_dump_knn"])
    else:
        knn_cp(X_dev, Y_dev, params["model_dump_knn"])

############# SVM ###########
if parmas["is_svm"]>0:
    print("[INFO] building SVMs ...")
    if params["is_cv_req_svm"]>0:
        svm_cp_cv(X_dev, Y_dev, svm_grid, params["model_dump_svm"])
    else:
        svm_cp(X_dev, Y_dev, params["model_dump_svm"])


############# Decision Trees ###########
if params["is_tree"] >0:
    print("[INFO] building Decision Tree ...")
    if params["is_cv_req_tree"] >0:
        tree_cp_cv(X_dev, Y_dev, tree_grid, params["model_dump_Dtree"])
    else:
        tree_cp(X_dev, Y_dev, params["model_dump_Dtree"])

############# Random Forests ############
if params["is_rf"] >0:
    print ("[INFO] building Random Forests ...")
    if params["is_cv_req_rf"]  >0:
        rf_cp_cv(X_dev, Y_dev, grid["rf_grid"], params["model_dump_RF"])
    else:
        rf_cp(X_dev, Y_dev, params["model_dump_RF"])


############ Gradient Boosting Machines #############
if params["is_gbm"] >0:
    print ("[INFO] building GBM ...")
    if params["is_cv_req_gbm"] >0:
        gbm_cp_cv(X_dev, Y_dev, grid["gbm_grid"], params["model_dump_GBM"])

    else:
        gbm_cp(X_dev, Y_dev, params["model_dump_GBM"])


############# XGBOOST ################
if params("is_xgb") >0:
    print ("[INFO] building XGBOOST...")
    if params["is_cv_req_xgboost"] >0:
        xgb_cp_cv(X_dev, Y_dev, grid["xgb_grid"], params["model_dump_xgboost"])

    else:
        xgb_cp(X_dev, Y_dev, params["model_dump_xgboost"])


print ("[INFO] Finish Building models")
print (" The best model is xgboost with 92 percent accuracy")
