def gmlw_SVM(X_train, Y_train, N, C, gamma, cv, scoring='accuracy'):
    """C and gamma are each a list of parameters to be used for grid search
    N is the number of top parameters that you want.
    scoring is the model metric to be used"""
    import numpy as np
    import pickle
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
    print('Parameters prepared')
    return(final)
