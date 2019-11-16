import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import sklearn.metrics as metrics
from sklearn import cluster
import itertools
from . import graphFunctions as graph
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA

def naive_bayes_analyzes(X, y, labels, estimators, rskf, title_complement= '- ', average='binary'):
    accuracy = {}
    sensitivity = {}
    cnf_mtx = {}
    for clf in estimators:
        accuracy[clf] = 0
        sensitivity[clf] = 0
        cnf_mtx[clf] = np.zeros((len(labels), len(labels))).astype(int)
    
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for clf in estimators:
            estimators[clf].fit(X_train, y_train)
            prdY = estimators[clf].predict(X_test)
            accuracy[clf] += metrics.accuracy_score(y_test, prdY)
            sensitivity[clf] += metrics.recall_score(y_test, prdY, average=average)
            cnf_mtx[clf] += metrics.confusion_matrix(y_test, prdY, labels)
    for clf in estimators:
        accuracy[clf] /= rskf.get_n_splits()
        sensitivity[clf] /= rskf.get_n_splits()
        print("Accuracy for:", clf, ':', format(accuracy[clf], '.4f'))
        print("Sensitivity for:", clf, ':', format(sensitivity[clf], '.4f'))
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
        graph.plot_confusion_matrix(axs[0, 0], cnf_mtx[clf], labels, title_complement=(title_complement + clf))
        graph.plot_confusion_matrix(axs[0, 1], cnf_mtx[clf], labels, normalize=True, title_complement=(title_complement + clf))
        plt.tight_layout()
        plt.show()
    return accuracy, sensitivity

def knn(X, y, n, dist, rskf):
    acc = 0
    recall = 0
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn = KNeighborsClassifier(n_neighbors=n, metric=dist)
        knn.fit(X_train, y_train)
        prdY = knn.predict(X_test)
        acc += metrics.accuracy_score(y_test, prdY)
        recall += metrics.recall_score(y_test, prdY)
    acc /= rskf.get_n_splits()
    recall /= rskf.get_n_splits()
    return acc, recall

def knn_analyzes(X, y, nvalues, dist, rskf, title_complement = '', average='binary'):
    values = {}
    recall = {}
    for d in dist:
        values[d] = np.zeros(len(nvalues))
        recall[d] = np.zeros(len(nvalues))

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for d in dist:
            yvalues = []
            recall_values = []
            for n in nvalues:
                knn = KNeighborsClassifier(n_neighbors=n, metric=d)
                knn.fit(X_train, y_train)
                prdY = knn.predict(X_test)
                yvalues.append(metrics.accuracy_score(y_test, prdY))
                recall_values.append(metrics.recall_score(y_test, prdY, average=average))
            values[d] += yvalues
            recall[d] += recall_values
    for d in dist:
        values[d] /= rskf.get_n_splits()
        recall[d] /= rskf.get_n_splits()
    plt.figure()
    graph.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants ' + title_complement, 'n', 'accuracy', percentage=True)
    plt.show()
    for aux in range(len(values['manhattan'])):
        print('Accuracy for n equal to', nvalues[aux], ':', format(values['manhattan'][aux], '.4f'))
        print('Sensitivity for n equal to', nvalues[aux], ':', format(recall['manhattan'][aux], '.4f'), '\n')
    return values, recall

def decision_tree_analyzes(X, y, min_samples_leaf, max_depths, criteria, rskf, average='binary'):
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
                    recall.append(metrics.recall_score(y_test, prdY, average=average))
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

def gradient_boosting(X, y, rskf, *, learning_rate=0.1, n_estimators=100, max_depth=3, max_features=None):
    accuracy = 0
    recall = 0

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        boost = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
        boost.fit(X_train, y_train)
        prdY = boost.predict(X_test)
        accuracy += metrics.accuracy_score(y_test, prdY)
        recall += metrics.recall_score(y_test, prdY)
    accuracy /= rskf.get_n_splits()
    recall /= rskf.get_n_splits()
    
    return accuracy, recall

def gradient_boosting_analyzes(X, y, range_variable, range_variable_name, rskf, *, learning_rate=0.1, n_estimators=100, max_depth=3, max_features=None):
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

def apriori_rules(dummified_df, minpaterns=10, minconf=0.9, minlengthrule=2, minsup=1):
    frequent_itemsets = {}
    while minsup>0:
        minsup = minsup*0.9
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
            rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
            if (len(rules[(rules['antecedent_len']>=minlengthrule)]) >= minpaterns):
                break
    patterns = len(frequent_itemsets)
    bigger_patterns = len(rules[(rules['antecedent_len']>=2)])
    avg_support = rules[(rules['antecedent_len']>=2)]['support'].mean()
    avg_confidence = rules[(rules['antecedent_len']>=2)]['confidence'].mean()
    avg_leverage = rules[(rules['antecedent_len']>=2)]['leverage'].mean()
    avf_lift = rules[(rules['antecedent_len']>=2)]['lift'].mean()
    return [patterns, bigger_patterns, avg_support, avg_confidence, avg_leverage, avf_lift], rules[(rules['antecedent_len']>=minlengthrule)]

def pattern_mining_analyzes(data, df_target, num_features, num_bins, metrics):
    y: np.ndarray = df_target.values
    X: np.ndarray = data.values

    results_cut = np.zeros((len(num_features), len(num_bins), len(metrics)))
    results_qcut = np.zeros((len(num_features), len(num_bins), len(metrics)))

    for k in range(len(num_features)):
        for bins in range(len(num_bins)):
            selector = SelectKBest(f_classif, k=num_features[k])
            X_new = selector.fit_transform(X, y)

            cols = selector.get_support(indices=True)

            selected_df = data.iloc[:, cols].join(df_target)
            selected_df.head()
            newdf_cut = selected_df.copy()
            newdf_qcut = selected_df.copy()
            for col in newdf_cut:
                if col not in ['class']: 
                    ## Discretize according to the size of the bins
                    newdf_cut[col] = pd.cut(newdf_cut[col],num_bins[bins],labels=range(num_bins[bins]))
                    ## Discretize according to the number of the elements per bins (similar to quartiles)
                    newdf_qcut[col] = pd.qcut(newdf_qcut[col],num_bins[bins],labels=range(num_bins[bins]))
            dummylist_cut = []
            dummylist_qcut = []
            for att in newdf_cut:
                if att in ['class', 'gender']: newdf_cut[att] = newdf_cut[att].astype('category')
                if att in ['class', 'gender']: newdf_qcut[att] = newdf_qcut[att].astype('category')
                dummylist_cut.append(pd.get_dummies(newdf_cut[[att]]))
                dummylist_qcut.append(pd.get_dummies(newdf_qcut[[att]]))
            dummified_df_cut = pd.concat(dummylist_cut, axis=1)
            dummified_df_qcut = pd.concat(dummylist_qcut, axis=1)

            results_cut[k, bins], rules_cut = apriori_rules(dummified_df_cut)
            results_qcut[k, bins], rules_qcut = apriori_rules(dummified_df_qcut)

    plot_results_cut = {}
    plot_results_qcut = {}
    for m in range(len(metrics)):
        plot_results_cut[m] = {}
        plot_results_qcut[m] = {}
        for n in range(len(num_bins)):
            plot_results_cut[m][num_bins[n]] = results_cut[:, n, m]
            plot_results_qcut[m][num_bins[n]] = results_qcut[:, n, m]

    fig, axs = plt.subplots(len(metrics), 2, figsize=(len(metrics)*3, 26))
    plt.subplots_adjust(hspace=0.3)
    for m in range(len(metrics)):
        graph.multiple_line_chart(axs[m, 0], num_features, plot_results_cut[m], "Cut - " + metrics[m] + " by number of features and number of bins", "Number of features", metrics[m])
        graph.multiple_line_chart(axs[m, 1], num_features, plot_results_qcut[m], "Qcut - " + metrics[m] + " by number of features and number of bins", "Number of features", metrics[m])
    return results_cut, results_qcut

def kmeans(X, y, n_clusters, y_interval=(0, 2100), y_interval2=(0, 0.35)):
    results = np.zeros((len(n_clusters), 5))
    for i in range(len(n_clusters)):
        kmeans = cluster.KMeans(n_clusters=n_clusters[i], random_state=1).fit(X)
        y_pred = kmeans.labels_
        value_counts = pd.Series(y_pred).value_counts()
        results[i][0] = kmeans.inertia_
        results[i][1] = metrics.silhouette_score(X, y_pred)
        results[i][2] = value_counts[value_counts == 1].shape[0]
        results[i][3] = metrics.adjusted_rand_score(y, y_pred)
        results[i][4] = metrics.homogeneity_score(y, y_pred)

    plt.figure()
    graph.double_line_chart_different_scales(plt.gca(), n_clusters, results[:, 0], results[:, 1], "SSD and Silhouette in diffent number of clusters", "Number of clusters", "Sum of squared distances", "Silhouette", y_interval=y_interval, y_interval2=y_interval2)
    return results

def visualize_cluster_PCA(X, y, n_plot):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(len(n_plot), 2, figsize=(14, 18))
    plt.subplots_adjust(hspace = 0.3)
    for i in range(len(n_plot)):
        kmeans = cluster.KMeans(n_clusters=n_plot[i], random_state=1).fit(X)
        y_pred = kmeans.labels_
        x_new = np.concatenate((x_pca, y.reshape((len(y), 1)), y_pred.reshape((len(y), 1))), axis=1)

        ax[i, 0].set_title("Real classes with PCA's axis")
        ax[i, 0].set_xlabel("PCA 1")
        ax[i, 0].set_ylabel("PCA 2")
        for j in range(len(np.unique(y))):
            ax[i, 0].scatter(x_new[x_new[:, 2] == j][:, 0], x_new[x_new[:, 2] == j][:, 1])

        ax[i, 1].set_title(str(n_plot[i])+ " clusters with PCA's axis")
        ax[i, 1].set_xlabel("PCA 1")
        ax[i, 1].set_ylabel("PCA 2")
        for j in range(len(np.unique(y_pred))):
            ax[i, 1].scatter(x_new[x_new[:, 3] == j][:, 0], x_new[x_new[:, 3] == j][:, 1])