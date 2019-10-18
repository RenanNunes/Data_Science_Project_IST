import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
import itertools
import graphFunctions as graph

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

def decision_tree_analyzes(X, y, min_samples_leaf, max_depths, criteria, rskf):
    accuracy = {}
    sensitivity = {}

    for c in criteria:
        accuracy[c] = {}
        sensitivity[c] = {}
        for max_d in max_depths:
            accuracy[c][max_d] = np.zeros((len(min_samples_leaf)))
            sensitivity[c][max_d] = np.zeros((len(min_samples_leaf)))

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for c in range(len(criteria)):
            crit = criteria[c]
            for d in max_depths:
                yvalues = []
                recall = []
                for n in min_samples_leaf:
                    tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=crit)
                    tree.fit(X_train, y_train)
                    prdY = tree.predict(X_test)
                    yvalues.append(metrics.accuracy_score(y_test, prdY))
                    recall.append(metrics.recall_score(y_test, prdY))
                accuracy[crit][d] += yvalues
                sensitivity[crit][d] += recall

    plt.figure()
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)

    for c in range(len(criteria)):
        for max_d in max_depths:
            accuracy[criteria[c]][max_d] /= rskf.get_n_splits()
            sensitivity[criteria[c]][max_d] /= rskf.get_n_splits()
        graph.multiple_line_chart(axs[0, c], min_samples_leaf, accuracy[criteria[c]], 'Decision Trees with %s criteria'%criteria[c], 'nr estimators', 'accuracy', percentage=True)
        graph.multiple_line_chart(axs[1, c], min_samples_leaf, sensitivity[criteria[c]], 'Decision Trees with %s criteria'%criteria[c], 'nr estimators', 'sensitivity', percentage=True)
    plt.show()
    return accuracy, sensitivity

def get_max_accuracy_sensitivity_data(accuracy, sensitivity, max_depths, min_samples_leaf, criteria):
    def get_max(main_data, second_data, max_depths, min_samples_leaf, criteria):
        max_row = []
        for d in max_depths:
            max_row.append(max(main_data[criteria][d]))
        max_data = max(max_row)
        max_depth = max_depths[max_row.index(max(max_row))]
        index_min_sample_leaf = np.where(main_data[criteria][max_depth] == max_data)[0][0]
        min_sample_leaf = min_samples_leaf[index_min_sample_leaf]
        return max_data, second_data[criteria][max_depth][index_min_sample_leaf], max_depth, min_sample_leaf
    max_acc = {}
    max_acc['acc'], max_acc['sens'], max_acc['max_depth'], max_acc['min_sample_leaf'] = get_max(accuracy, sensitivity, max_depths, min_samples_leaf, criteria)
    max_sens = {}
    max_sens['sens'], max_sens['acc'], max_sens['max_depth'], max_sens['min_sample_leaf'] = get_max(sensitivity, accuracy, max_depths, min_samples_leaf, criteria)
    return max_acc, max_sens
