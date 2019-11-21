import sys
import ast
import functools as ft
import pandas as pd
import exercise.pdreport as pdreport
import exercise.ctreport as ctreport


def report(source, dataframe, task):
    if source == 'PD':
        reportsource = pdreport
    elif source == 'CT':
        reportsource = ctreport
    else:
        return "Source error"

    if task == "preprocessing":
        return reportsource.preprocessing(dataframe)
    elif task == "unsupervised":
        return reportsource.unsupervised(dataframe)
    elif task == "classification":
        return reportsource.classification(dataframe)
    else:
        return "Task error"


def eval_input(total, elem):
    try:
        total.append(ast.literal_eval(elem))
    except:
        total.append(elem)
    return total


if __name__ == '__main__':

    '''A: read arguments'''
    args = sys.stdin.readline().rstrip('\n').split(' ')
    n, _source, _task = int(args[0]), args[1], args[2]
    
    '''B: read dataset'''
    data, header = [], sys.stdin.readline().rstrip('\n').split(',')
    for i in range(n-1):
        line = sys.stdin.readline().rstrip('\n').split(',')
        data.append(ft.reduce(eval_input, line, []))
    _dataframe = pd.DataFrame(data, columns=header)

    '''C: output results'''
    report(_source, _dataframe, _task)
