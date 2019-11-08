import pandas as pd


def preprocessing(dataframe: pd.DataFrame):
    report = ''
    report += '1.Data exploration\n'
    report += '1.1 Data Shape: ( ' + str(dataframe.shape[0]) + ' , ' + str(dataframe.shape[1]) + ' )\n'
    report += '1.2 Data Head: \n'
    # report += dataframe.head(10)
    return report


def unsupervised(dataframe: pd.DataFrame):
    return


def classification(dataframe: pd.DataFrame):
    return
