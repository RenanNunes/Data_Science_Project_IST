import numpy as np


def reject_variables(variables, reject=0.9):
    corr_mtx = variables.corr()
    rejected_corr = []
    for row in np.arange(corr_mtx.shape[0]):
        for column in np.arange(row+1, corr_mtx.shape[1]):
            if abs(corr_mtx.iloc[row, column]) > reject:
                rejected_corr.append(corr_mtx.columns.values.tolist()[column])
    return rejected_corr
