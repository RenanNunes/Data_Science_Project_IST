import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import BernoulliNB
import utils.dataPreparationFunctions as dataPrepare
import utils.modelAnalyzesFunctions as modelAnalyze


def preprocessing(dataframe: pd.DataFrame):
    print("1 Data exploration")
    # Shape
    print("1.1 Data Shape: %s" % (dataframe.shape,))
    # Head
    print("1.2 Data Head:")
    print(dataframe.head())
    # Type
    print("1.3 Data Distribuition:")
    print(dataframe.describe())
    print("1.3 Data Types:")
    print(dataframe.dtypes.value_counts())
    # Classes
    if "Cover_Type" in dataframe:
        print("1.4 Class count:")
        print(dataframe["Cover_Type"].value_counts())
    print()
    # Preprocessing
    _preprocess(dataframe, True, "2")
    return


def _preprocess(dataframe: pd.DataFrame, log: bool = False, log_number: str = "0"):
    count = 0
    if log:
        print(log_number + ".Data preparation")

    # Missing data
    missing_values = dataPrepare.get_missing_values(dataframe)
    if log:
        count += 1
        print(log_number + "." + str(count) + " Missing data")
        if missing_values[1] == 0:
            print("There are no missing data")
        else:
            print("There are missing data")  # TODO deal with missing data

    # Normalization
    cover_type = dataframe["Cover_Type"]
    dataframe = dataPrepare.normalize_data(dataframe.iloc[:, 0:dataframe.shape[1]-1])
    dataframe["Cover_Type"] = cover_type
    if log:
        count += 1
        print(log_number + "." + str(count) + " Normalization:")
        print(dataframe.describe())

    # Data balancing
    dataframe = dataPrepare.undersample(dataframe, "Cover_Type")
    if log:
        count += 1
        print(log_number + "." + str(count) + " Data balancing:")
        print(log_number + "." + str(count) + ".1 Data Shape: %s" % (dataframe.shape,))

    return dataframe, missing_values[1]


def classification(dataframe: pd.DataFrame):
    dataframe, missing_values = _preprocess(dataframe)
    print("1 Data Preprocessing:")
    print("There was %d missing values" % missing_values)
    print("The data was normalized")
    print("The data was undersampled by the minority class")

    print()
    print("2 Classifiers:")
    y = dataframe.pop("Cover_Type").values
    x = dataframe.values

    n_splits = 4
    n_repeats = 3
    rskf = RepeatedStratifiedKFold(n_splits, n_repeats)

    accuracy = {}
    sensitivity = {}
    confusion_matrix = {}

    # NB
    print("2.1 Naive Bayes:")
    print("2.1.1 Parameterization:")
    print("Estimator: Bernoulli")
    estimator = BernoulliNB()
    print("2.1.2 Results:")
    accuracy["NB"], sensitivity["NB"], confusion_matrix["NB"] = modelAnalyze.naive_bayes(x, y, estimator,
                                                                                         rskf, average='micro')
    print("Accuracy: %f" % accuracy["NB"])
    print("Sensitivity: %f" % sensitivity["NB"])
    print("Confusion Matrix:")
    print(confusion_matrix["NB"].round(3))
    print()

    # KNN
    print("2.2 KNN:")
    print("2.2.1 Parameterization:")
    print("Distance: Manhattan")
    distance = "manhattan"
    print("K: 1")
    k = 1
    print("2.2.2 Results:")
    accuracy["KNN"], sensitivity["KNN"], confusion_matrix["KNN"] = modelAnalyze.knn(x, y, k, distance,
                                                                                    rskf, average='micro')
    print("Accuracy: %f" % accuracy["KNN"])
    print("Sensitivity: %f" % sensitivity["KNN"])
    print("Confusion Matrix:")
    print(confusion_matrix["KNN"].round(3))
    print()

    # DT
    print("2.3 Decision Tree:")
    print("2.3.1 Parameterization:")
    print("Minimum number of samples: 0.001")
    min_samples = 0.001
    print("Maximum depth: 50")
    max_depth = 50
    print("Function: Entropy")
    criteria = "entropy"
    print("2.3.2 Results:")
    accuracy["DT"], sensitivity["DT"], confusion_matrix["DT"] = modelAnalyze.decision_tree(x, y, min_samples,
                                                                                           max_depth, criteria, rskf,
                                                                                           average='micro')
    print("Accuracy: %f" % accuracy["DT"])
    print("Sensitivity: %f" % sensitivity["DT"])
    print("Confusion Matrix:")
    print(confusion_matrix["DT"].round(3))
    print()

    # RF
    print("2.4 Random Forests:")
    print("2.4.1 Parameterization:")
    print("Number of estimators: 250")
    n_estimator = 250
    print("Maximum depth: 50")
    max_depth = 50
    print("Maximum feature: Sqrt")
    max_feature = 'sqrt'
    print("2.4.2 Results:")
    accuracy["RF"], sensitivity["RF"], confusion_matrix["RF"] = modelAnalyze.random_forest(x, y, rskf,
                                                                                           n_estimators=n_estimator,
                                                                                           max_depth=max_depth,
                                                                                           max_features=max_feature,
                                                                                           average='micro')
    print("Accuracy: %f" % accuracy["RF"])
    print("Sensitivity: %f" % sensitivity["RF"])
    print("Confusion Matrix:")
    print(confusion_matrix["RF"].round(3))
    print()

    # GB
    print("2.5 Gradient Boosting:")
    print("2.5.1 Parameterization:")
    print("Learning rate: 0.5")
    l_rate = 0.5
    print("Number of estimators: 300")
    n_estimator = 300
    print("Maximum depth: 10")
    max_depth = 10
    print("Maximum feature: Sqrt")
    max_feature = 'sqrt'
    print("2.5.2 Results:")
    accuracy["GB"], sensitivity["GB"], confusion_matrix["GB"] = modelAnalyze.gradient_boosting(x, y, rskf,
                                                                                               learning_rate=l_rate,
                                                                                               n_estimators=n_estimator,
                                                                                               max_depth=max_depth,
                                                                                               max_features=max_feature,
                                                                                               average='micro')
    print("Accuracy: %f" % accuracy["GB"])
    print("Sensitivity: %f" % sensitivity["GB"])
    print("Confusion Matrix:")
    print(confusion_matrix["GB"].round(3))
    print()

    # Comparative performance
    print("3 Comparative performance:")
    print(pd.DataFrame([pd.Series(accuracy), pd.Series(sensitivity)], ["Accuracy", "Sensitivity"]))
    return


def unsupervised(dataframe: pd.DataFrame):
    dataframe, missing_values = _preprocess(dataframe)
    print("1 Data Preprocessing:")
    print("There was %d missing values" % missing_values)
    print("The data was normalized")
    print("The data was undersampled by the minority class")

    print("2 Unsupervised:")
    y = dataframe.pop('Cover_Type').values
    x = dataframe.values

    # Clustering
    print("2.1 Clustering:")
    print("2.1.1 Parameterization:")
    print("Number of clusters: 3")
    n_cluster = 3
    print("Applying: K-Means")
    results = modelAnalyze.kmean(x, y, n_cluster)
    print("2.1.2 Results:")
    print("Inertia: %.4f" % results[0])
    print("Silhouette: %.4f" % results[1])
    print("Adjusted Rand: %.4f" % results[3])
    print("Homogeneity: %.4f" % results[4])

    # Pattern Mining
    # print("2.2 Pattern Mining:")
    # print("2.2.1 Parameterization:")
    # print("2.2.2 Results:")

    return
