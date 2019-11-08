import sys
import pandas as pd
import pdreport
import ctreport
# 10 PD preprocessing

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


if __name__ == '__main__':

    '''A: read arguments'''
    args = sys.stdin.readline().rstrip('\n').split(' ')
    n, _source, _task = int(args[0]), args[1], args[2]
    
    '''B: read dataset'''
    a = sys.stdin.readline()
    data, header = [], a.rstrip('\n').split(',')
    for i in range(n-1):
        data.append(sys.stdin.readline().rstrip('\n').split(','))    
    _dataframe = pd.DataFrame(data, columns=header)

    '''C: output results'''
    print(report(_source, _dataframe, _task))
