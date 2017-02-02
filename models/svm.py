import glob, os
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV


"""
Two ways of training the model :
 - Perform Search (Grid/Random) and train the model on top three results.
 - Directly train the model without any model.


"""

def svm_cp(X_dev, Y_dev, dump_file):
    """
    create a model (model_dump_svm.pkl) in models_dump folder.
    X_dev = input data
    Y_dev = target data
    dump_file = file name to store model

    to test:
    from sklearn import datasets
    iris = datasets.load_iris()
    svm_cp(iris.data, iris.target,"pickle.pkl")
    """
    clf = svm.SVC()
    svm_fit = clf.fit(X_dev, Y_dev)
    with open(dump_file, "wb") as f:
        pickle.dump(svm_fit,f)


def svm_cp_cv(X_dev, Y_dev, param_grid, dump_file, cv = 5, score = "accuracy" , n_best = 3, search = 0):
    """
    create a model (model_dump_svm.pkl) in models_dump folder.
    X_dev: input data
    Y_dev: target data
    param_grid: A dictionary object to tell the best params.
    dump_file: file name to store model
    cv: no_of folds for cross validation, def00ault is 5
    score: A score to select the best params0.
    n_best: number of top models to store
    search: A binary input which performs random search if 0 (default) and Grid search otherwise.

    Returns:
    A pickle file which has the n_best models.

    to test :
    from sklearn import datasets
    iris = datasets.load_iris()
    svm_cp_cv(iris.data, iris.target, tuned_parameters, "pickle.pkl", cv = 5, score = "precision_macro" , n_best = 3, search = 1)
    """
    clf = svm.SVC(X_dev, Y_dev)
    if search == 0:
        pass
    else:
        grid_search = GridSearchCV(c, param_grid = param_grid, cv = cv, scoring = score)
        grid_search.fit(X_dev, Y_dev)#perform the search
        mean_test_score = grid_search.cv_results_["mean_test_score"]
        results = grid_search.cv_results_["params"]
        final = [(x, y) for (y, x) in sorted(zip(mean_test_score, results), key=lambda pair: pair[0], reverse=True)][:n_best]
        models = {}
        for i in range(len(final)):
            models["model_{}".format(i)] = svm.SVC(X_dev, Y_dev,final[i][0])

        with open(dump_file,"wb") as f:
            pickle.dump(models,f)
