def gmlw_SVM_cv(X_train, Y_train, N, C, gamma, cv, scoring='accuracy'):
    """This function returns the parameters of top N models based on the scoring parameter.
    X_train and Y_train are the training data and the response variable respectively.
    C and gamma are each a list of parameters to be used for grid search.
    N is the number of models whose parameters you want.
    cv is the number of cross validation folds.
    scoring is the model metric to be used"""
    import numpy as np
#     import pickle
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    param_grid = {'C': C,
             'gamma': gamma}
    grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)
    mean_test_score = grid_search.cv_results_['mean_test_score']
    sorted_mean_test_score = np.argsort(mean_test_score)
    results = grid_search.cv_results_['params']
    final = [(x, y) for (y, x) in sorted(zip(mean_test_score, results), key=lambda pair: pair[0], reverse=True)][:N]
    print('Parameters prepared.')
    return(final)


def gmlw_SVM(svm_ml, X_train, y_train):
    import pickle
    parameters = [i[0] for i in svm_ml]
    svm_models = []
    for i in parameters:
        svm_cost = i['C']
        svm_gamma = i['gamma']
        svm_model = SVC(C=svm_cost, gamma=svm_gamma)
        svm_model.fit(X_train, y_train)
        svm_models.append(svm_model)
    with open('SVM_demo.pkl', 'wb') as f:
        pickle.dump(svm_models, f)
    """The models are dumped in a single pickle file called SVM_demo.pkl"""
    print('Models pickled.')
    return(L)
