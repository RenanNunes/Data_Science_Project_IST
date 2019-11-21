import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
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
    if "class" in dataframe:
        print("1.4 Class count:")
        print(dataframe["class"].value_counts())
    print()
    # Preprocessing
    _preprocess(dataframe, True, "2")


def _preprocess(dataframe: pd.DataFrame, log: bool = False, log_number: str = "0", *, correlation: float = 0.95):
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

    # Multiple data from the same ID
    dataframe = dataframe.groupby(dataframe.id).mean()
    if log:
        count += 1
        print(log_number + "." + str(count) + " Duplicate IDs:")
        print("Grouping all instances with the same id by the mean value")
        print(log_number + "." + str(count) + ".1 Data Head:")
        print(dataframe.head())
        print(log_number + "." + str(count) + ".2 Data Shape: %s" % (dataframe.shape,))

    # Normalization
    dataframe = dataPrepare.normalize_data(dataframe)
    if log:
        count += 1
        print(log_number + "." + str(count) + " Normalization:")
        print(dataframe.describe())

    # Data balancing
    dataframe, smote = dataPrepare.oversample(dataframe, "class")
    if log:
        count += 1
        print(log_number + "." + str(count) + " Data balancing:")
        print(log_number + "." + str(count) + ".1 Data Shape: %s" % (dataframe.shape,))
        print(log_number + "." + str(count) + ".2 Smote result:")
        print(smote)

    # Feature selection
    dataframe = dataPrepare.reject_variables(dataframe, correlation)  # TODO change the correlation treshold
    if log:
        count += 1
        print(log_number + "." + str(count) + " Feature selection:")
        print("All variables with more then %.2f correlation were simplified into one variable" % correlation)
        print(log_number + "." + str(count) + ".1 Data Shape: %s" % (dataframe.shape,))
        print(log_number + "." + str(count) + ".2 Data head:")
        print(dataframe.head())

    return dataframe, missing_values[1], smote, correlation


def classification(dataframe: pd.DataFrame):
    dataframe, missing_values, smote, correlation = _preprocess(dataframe)
    print("1 Data Preprocessing:")
    print("There was %d missing values" % missing_values)
    print("The data was grouped by id")
    print("The data was normalized")
    print("Smote was applied to the minority class")
    print("All variables with more than %.2f correlation were simplified into one variable" % correlation)

    print()
    print("2 Classifiers:")
    y = dataframe.pop('class').values
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
    print("Estimator: Multinomial")
    estimator = MultinomialNB()
    print("2.1.2 Results:")
    accuracy["NB"], sensitivity["NB"], confusion_matrix["NB"] = modelAnalyze.naive_bayes(x, y, estimator, rskf)
    print("Accuracy: %f" % accuracy["NB"])
    print("Sensitivity: %f" % sensitivity["NB"])
    print("Confusion Matrix:")
    print(confusion_matrix["NB"])
    print()

    # KNN
    print("2.2 KNN:")
    print("2.2.1 Parameterization:")
    print("Distance: Manhattan")
    distance = "manhattan"
    print("K: 1")
    k = 1
    print("2.2.2 Results:")
    accuracy["KNN"], sensitivity["KNN"], confusion_matrix["KNN"] = modelAnalyze.knn(x, y, k, distance, rskf)
    print("Accuracy: %f" % accuracy["KNN"])
    print("Sensitivity: %f" % sensitivity["KNN"])
    print("Confusion Matrix:")
    print(confusion_matrix["KNN"])
    print()

    # DT
    print("2.3 Decision Tree:")
    print("2.3.1 Parameterization:")
    print("Minimum number of samples: 0.001")
    min_samples = 0.001
    print("Maximum depth: 50")
    max_depth = 50
    print("Function: Gini")
    criteria = "gini"
    print("2.3.2 Results:")
    accuracy["DT"], sensitivity["DT"], confusion_matrix["DT"] = modelAnalyze.decision_tree(x, y, min_samples,
                                                                                           max_depth, criteria, rskf)
    print("Accuracy: %f" % accuracy["DT"])
    print("Sensitivity: %f" % sensitivity["DT"])
    print("Confusion Matrix:")
    print(confusion_matrix["DT"])
    print()

    # RF
    print("2.4 Random Forests:")
    print("2.4.1 Parameterization:")
    print("Number of estimators: 300")
    n_estimator = 300
    print("Maximum depth: 25")
    max_depth = 25
    print("Maximum feature: Sqrt")
    max_feature = 'sqrt'
    print("2.4.2 Results:")
    accuracy["RF"], sensitivity["RF"], confusion_matrix["RF"] = modelAnalyze.random_forest(x, y, rskf,
                                                                                           n_estimators=n_estimator,
                                                                                           max_depth=max_depth,
                                                                                           max_features=max_feature)
    print("Accuracy: %f" % accuracy["RF"])
    print("Sensitivity: %f" % sensitivity["RF"])
    print("Confusion Matrix:")
    print(confusion_matrix["RF"])
    print()

    # GB
    print("2.5 Gradient Boosting:")
    print("2.5.1 Parameterization:")
    print("Learning rate: 0.2")
    l_rate = 0.2
    print("Number of estimators: 200")
    n_estimator = 200
    print("Maximum depth: 3")
    max_depth = 3
    print("Maximum feature: Log2")
    max_feature = 'log2'
    print("2.5.2 Results:")
    accuracy["GB"], sensitivity["GB"], confusion_matrix["GB"] = modelAnalyze.gradient_boosting(x, y, rskf,
                                                                                               learning_rate=l_rate,
                                                                                               n_estimators=n_estimator,
                                                                                               max_depth=max_depth,
                                                                                               max_features=max_feature)
    print("Accuracy: %f" % accuracy["GB"])
    print("Sensitivity: %f" % sensitivity["GB"])
    print("Confusion Matrix:")
    print(confusion_matrix["GB"])
    print()

    # Comparative performance
    print("3 Comparative performance:")
    print(pd.DataFrame([pd.Series(accuracy), pd.Series(sensitivity)], ["Accuracy", "Sensitivity"]))
    return


def unsupervised(dataframe: pd.DataFrame):
    dataframe, missing_values, smote, correlation = _preprocess(dataframe)
    print("1 Data Preprocessing:")
    print("There was %d missing values" % missing_values)
    print("The data was grouped by id")
    print("The data was normalized")
    print("Smote was applied to the minority class")
    print("All variables with more than %.2f correlation were simplified into one variable" % correlation)

    print("2 Unsupervised:")

    # Pattern Mining
    print("2.1 Clustering:")
    print("2.1.1 Parameterization:")
    print("2.1.2 Results:")

    # Clustering
    print("2.2 Clustering:")
    print("2.2.1 Parameterization:")
    print("Number of clusters: 5")
    print("Applying: K-Means")
    print("2.2.2 Results:")

    return
