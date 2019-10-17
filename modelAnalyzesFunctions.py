import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn.metrics as metrics
import itertools

def naive_bayes_analyzes(X, y, labels, estimators, rskf, title_complement= '- '):
    accuracy = {}
    sensitivity = {}
    cnf_mtx = {}
    for clf in estimators:
        accuracy[clf] = 0
        sensitivity[clf] = 0
        cnf_mtx[clf] = np.zeros((2, 2)).astype(int)
    
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for clf in estimators:
            estimators[clf].fit(X_train, y_train)
            prdY = estimators[clf].predict(X_test)
            accuracy[clf] += metrics.accuracy_score(y_test, prdY)
            sensitivity[clf] += metrics.recall_score(y_test, prdY)
            cnf_mtx[clf] += metrics.confusion_matrix(y_test, prdY, labels)
    for clf in estimators:
        accuracy[clf] /= n_splits*n_repeats
        sensitivity[clf] /= n_splits*n_repeats
        print("Accuracy for:", clf, ':', format(accuracy[clf], '.4f'))
        print("Sensitivity for:", clf, ':', format(sensitivity[clf], '.4f'))
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
        graph.plot_confusion_matrix(axs[0, 0], cnf_mtx[clf], labels, title_complement=(title_complement + clf))
        graph.plot_confusion_matrix(axs[0, 1], cnf_mtx[clf], labels, normalize=True, title_complement=(title_complement + clf))
        plt.tight_layout()
        plt.show()
    return accuracy, sensitivity