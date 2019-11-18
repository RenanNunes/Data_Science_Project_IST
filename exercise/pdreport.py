import pandas as pd


def preprocessing(dataframe: pd.DataFrame):
    print("1.Data exploration")
    # Shape
    print("1.1 Data Shape: %s" % (dataframe.shape,))
    # Head
    print("1.2 Data Head:")
    print(dataframe.head())
    # Type
    print("1.3 Data Types:")
    print(dataframe.dtypes.value_counts())
    # Classes
    if "class" in dataframe or "d" in dataframe:
        print("1.4 Class count:")
        print(dataframe["class"].value_counts())





def unsupervised(dataframe: pd.DataFrame):
    return


def classification(dataframe: pd.DataFrame):
    return
