"""utils
This script allows calculate word similarities and save in python dictionaries
"""
import datetime
from termcolor import colored

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

def show_content_file(filename, lines=5, in_color=True):
    """
    Prints names of top n results to file called filename in accordance with out result file format.
    See README for more details.
    :param n: integer
    :param filename: string
    :return: None
    """
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            if lines == 0: break   ## counter
            if in_color:
                line = line[:-1].split(" ")
                print(f"{colored('t', 'green')} {colored(line[0], 'blue')} {colored(line[1], 'blue')}")
            else:
                print(f"{line}")
            lines = lines - 1
