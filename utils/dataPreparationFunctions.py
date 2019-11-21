import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


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


def oversample(data, target_attribute, RANDOM_STATE = 42, RATIO = 'minority'):
    smote = SMOTE(ratio=RATIO, random_state=RANDOM_STATE)
    data_columns = data.columns
    y = data.pop(target_attribute).values
    x = data.values
    smote_x, smote_y = smote.fit_sample(x, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    oversampled = pd.concat([pd.DataFrame(smote_x), pd.DataFrame(smote_y)], axis=1)
    oversampled.columns = data_columns
    oversampled.index.name = 'id'
    return oversampled, smote_target_count


def undersample(data, target_attribute):
    under_sampler = RandomUnderSampler(random_state=42)
    y: np.ndarray = data.pop(target_attribute).values
    x: np.ndarray = data.values
    x_res, y_res = under_sampler.fit_resample(x, y)
    return pd.concat([pd.DataFrame(x_res), pd.DataFrame(y_res)], axis=1)
