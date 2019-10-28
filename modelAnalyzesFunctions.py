import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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
    decision_trees = {}

    for c in criteria:
        accuracy[c] = {}
        sensitivity[c] = {}
        decision_trees[c] = {}
        for max_d in max_depths:
            accuracy[c][max_d] = np.zeros((len(min_samples_leaf)))
            sensitivity[c][max_d] = np.zeros((len(min_samples_leaf)))
            decision_trees[c][max_d] = []

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for crit in criteria:
            trees = []
            for d in max_depths:
                yvalues = []
                recall = []
                for n in min_samples_leaf:
                    tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=crit)
                    tree.fit(X_train, y_train)
                    prdY = tree.predict(X_test)
                    yvalues.append(metrics.accuracy_score(y_test, prdY))
                    recall.append(metrics.recall_score(y_test, prdY))
                    trees.append(tree)
                accuracy[crit][d] += yvalues
                sensitivity[crit][d] += recall
                decision_trees[crit][d] = trees

    plt.figure()
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)

    for c in range(len(criteria)):
        for max_d in max_depths:
            accuracy[criteria[c]][max_d] /= rskf.get_n_splits()
            sensitivity[criteria[c]][max_d] /= rskf.get_n_splits()
        graph.multiple_line_chart(axs[0, c], min_samples_leaf, accuracy[criteria[c]], 'Decision Trees with %s criteria'%criteria[c], 'min samples leaf ', 'accuracy')
        graph.multiple_line_chart(axs[1, c], min_samples_leaf, sensitivity[criteria[c]], 'Decision Trees with %s criteria'%criteria[c], 'min samples leaf', 'sensitivity')
    plt.show()
    return accuracy, sensitivity, decision_trees

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

def random_forest(X, y, rskf, *, n_estimators=10, max_depth=None, max_features="auto"):
    accuracy = 0
    sensitivity = 0

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        rf.fit(X_train, y_train)
        prdY = rf.predict(X_test)
        accuracy += metrics.accuracy_score(y_test, prdY)
        sensitivity += metrics.recall_score(y_test, prdY)
    accuracy /= rskf.get_n_splits()
    sensitivity /= rskf.get_n_splits()
    
    return accuracy, sensitivity

def random_forest_analyzes(X, y, range_variable, range_variable_name, rskf, *, n_estimators=10, max_depth=None, max_features="auto"):
    accuracy = {}
    sensitivity = {}
    for variable in range_variable:
        accuracy[variable] = 0
        sensitivity[variable] = 0
    
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for variable in range_variable:  
            param = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features,
                range_variable_name: variable,
            }
            rf = RandomForestClassifier(**param)
            rf.fit(X_train, y_train)
            prdY = rf.predict(X_test)
            
            accuracy[variable] += metrics.accuracy_score(y_test, prdY)
            sensitivity[variable] += metrics.recall_score(y_test, prdY)
    
    for variable in range_variable:
        accuracy[variable] /= rskf.get_n_splits()
        sensitivity[variable] /= rskf.get_n_splits()
    
    plt.figure()
    graph.double_line_chart_different_scales(plt.gca(), range_variable, accuracy.values(), sensitivity.values(), 'Random Forests with different %s'%range_variable_name, range_variable_name, 'accuracy', 'sensitivity', y_interval=(0.75, 0.95), y_interval2=(0.75, 0.95))
    plt.show()
    
    return accuracy, sensitivity

def gradient_boosting(X, y, rskf):
    accuracy = 0
    recall = 0

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        boost = GradientBoostingClassifier()
        boost.fit(X_train, y_train)
        prdY = boost.predict(X_test)
        accuracy += metrics.accuracy_score(y_test, prdY)
        recall += metrics.recall_score(y_test, prdY)
    accuracy /= rskf.get_n_splits()
    recall /= rskf.get_n_splits()
    
    return accuracy, recall

def gradiente_boosting_analyzes(X, y, range_variable, range_variable_name, rskf, *, learning_rate=0.1, n_estimators=100, max_depth=3, max_features=None):
    accuracy = {}
    recall = {}
    for variable in range_variable:
        accuracy[variable] = 0
        recall[variable] = 0

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for variable in range_variable:
            parameters = {
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'max_features': max_features,
                range_variable_name: variable
            }
            boost = GradientBoostingClassifier(**parameters)
            boost.fit(X_train, y_train)
            prdY = boost.predict(X_test)
            accuracy[variable] += metrics.accuracy_score(y_test, prdY)
            recall[variable] += metrics.recall_score(y_test, prdY)

    for variable in range_variable:
        accuracy[variable] /= rskf.get_n_splits()
        recall[variable] /= rskf.get_n_splits()

    plt.figure()
    graph.double_line_chart_different_scales(plt.gca(), range_variable, accuracy.values(), recall.values(), 'Gradient Boosting with different %s'%range_variable_name, range_variable_name, 'accuracy', 'sensitivity', y_interval=(0.75, 0.95), y_interval2=(0.75, 0.95))
    plt.show()
    
    return accuracy, recall
