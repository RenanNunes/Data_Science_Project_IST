import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as _stats
import numpy as np
import itertools


def choose_grid(nr):
    return (nr+3) // 4, 4


def line_chart(ax: plt.Axes, series: pd.Series, title: str, xlabel: str, ylabel: str, percentage=False, y_interval=(-1, -1)):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    elif y_interval != (-1, -1):
        ax.set_ylim(y_interval)
    ax.plot(series)


def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False, y_interval=(-1, -1)):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    elif y_interval != (-1, -1):
        ax.set_ylim(y_interval)
    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox = True, shadow = True)

def double_line_chart_different_scales(ax: plt.Axes, xvalues: list, yvalues: pd.Series, yvalues2: pd.Series, title: str, xlabel: str, ylabel: str, ylabel2: str, percentage=False, y_interval=(-1, -1), percentage2=False, y_interval2=(-1,-1)):
    ax2 = ax.twinx()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color='darkblue')
    ax2.set_ylabel(ylabel2, color='darkorange')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    elif y_interval != (-1, -1):
        ax.set_ylim(y_interval)
    if percentage2:
        ax2.set_ylim(0.0, 1.0)
    elif y_interval2 != (-1, -1):
        ax2.set_ylim(y_interval2)
        
    ax.plot(xvalues, yvalues, color='darkblue')
    ax2.plot(xvalues, yvalues2, color='darkorange')

def bar_chart(ax: plt.Axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False, y_interval=(-1, -1)):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=0, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    elif y_interval != (-1, -1):
        ax.set_ylim(y_interval)
    ax.bar(xvalues, yvalues, edgecolor='grey')


def multiple_bar_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False, y_interval=(-1, -1)):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    elif y_interval != (-1, -1):
        ax.set_ylim(y_interval)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(loc='lower center', ncol=len(yvalues), bbox_to_anchor=(0.5, -0.2), fancybox = True, shadow = True)


def compute_known_distributions(x_values, n_bins, normal, logNorm, exp, skewNorm) -> dict:
    distributions = dict()
    # Gaussian
    if (normal):
        mean, sigma = _stats.norm.fit(x_values)
        distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    if (logNorm):
        sigma, loc, scale = _stats.lognorm.fit(x_values)
        distributions['LogNorm(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    if (exp):
        loc, scale = _stats.expon.fit(x_values)
        distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    if (skewNorm):
        a, loc, scale = _stats.skewnorm.fit(x_values)
        distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions


def histogram(ax: plt.Axes, series: pd.Series, title = '', xlabel = '', ylabel = '', bins = 10):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.hist(series, bins)
        

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, title = '', xlabel = '', ylabel = '', normal = True, logNorm = False, exp = True, skewNorm = False):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 20, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins, normal, logNorm, exp, skewNorm)
    multiple_line_chart(ax, values, distributions, title, xlabel, ylabel)
    
    
def histogram_with_two_classes(ax: plt.Axes, series_1: pd.Series, series_2: pd.Series, title = '', xlabel = '', ylabel = '', label_classes = ['0', '1'], density = True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.hist([series_1, series_2], 'auto', density=density, label=label_classes)
    ax.legend(loc='best')


def scatter_with_two_classes(ax: plt.Axes, data_1: pd.Series, data_2: pd.Series, variable_1: str, variable_2: str):
        ax.set_title("%s x %s"%(variable_1, variable_2))
        ax.set_xlabel(variable_1)
        ax.set_ylabel(variable_2)
        ax.scatter(data_1[variable_1], data_1[variable_2])
        ax.scatter(data_2[variable_1], data_2[variable_2])
        
def plot_confusion_matrix(ax: plt.Axes, cnf_matrix: np.ndarray, classes_names: list, normalize: bool = False, title_complement = ''):
    CMAP = plt.cm.Blues
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / total
        title = 'Normalized confusion matrix ' + title_complement
    else:
        cm = cnf_matrix
        title = 'Confusion matrix ' + title_complement
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=CMAP)

    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")
