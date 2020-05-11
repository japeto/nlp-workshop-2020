"""utils
This script allows calculate word similarities and save in python dictionaries
"""
import datetime
import numpy as np

def flatten(list):
    """
    Return a list of a level
    :param lst: list with list inside
    :return: list
    """
    return [elem for sublst in list for elem in sublst]

def list_softmax(lst):
    """
    Normalizes list numeric values using softmax
    :param lst: numbers
    :return: list
    """
    return np.exp(lst) / np.sum(np.exp(lst), axis=0)

def list_normalize(lst):
    """
    Normalizes list numeric values
    :param lst: numbers
    :return: list
    """
    return [elem / sum(lst) for elem in lst]

def save_stats(vec, full=True):
    """
    Save statistics - median, std, mean - about given vector
    :param vec: list of numbers
    :param full: True or False
    :return: None
    """
    median = float(np.median(vec))
    std = float(np.std(vec))
    mean = float(np.mean(vec))
    output = (median, std, mean)
    with open(f'outputs/stats_{date_object}.txt','a') as f:
        if full:
            f.write("median: %f\tstd: %f\tmean: %f" % output)
        else:
            f.write("%f\t%f\t%f" % output)
        f.write('\n')
