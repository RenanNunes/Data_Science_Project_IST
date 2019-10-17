import numpy as np


def rejectable_variables(data, reject_treshold=0.9):
    corr_mtx = data.corr()
    rejected_corr = []
    for row in np.arange(corr_mtx.shape[0]):
        for column in np.arange(row+1, corr_mtx.shape[1]):
            if abs(corr_mtx.iloc[row, column]) > reject_treshold:
                rejected_corr.append(corr_mtx.columns.values.tolist()[column])
    return rejected_corr

def reject_variables(data, reject_treshold=0.9):
    return data.loc[:, ~data.columns.isin(rejectable_variables(data, reject_treshold))]

def get_missing_values(data):
    mv = {}
    count_na = 0
    for var in data:
        mv[var] = data[var].isna().sum()
        count_na += data[var].isna().sum()
    return mv, count_na

def normalize_data(data):
    return (data-data.min())/(data.max()-data.min())
