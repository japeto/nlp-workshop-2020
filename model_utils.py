from scipy.spatial import distance
import numpy as np
from gensim.models import Word2Vec
import os
import time
import random
import re
import glob
import datetime
import math
from collections import Counter

# our files
from helper import *
import config
from result_file import ResultList

def svm_file_name(set_name, timestamp, file_type, svm_folder="outputs/svm/"):
    """
    Creates filename for SVM file. set_name is name of given seed set, e.g. capitals.
    timestamp is an time identifier of created sef of SVM files. file_type is a type
    of SVM file created (train, test, model, prediction)
    :param set_name: string
    :param timestamp: integer
    :param file_type: string
    :return: string
    """
    return "%s%s-%s-%s-svm" % (svm_folder, set_name, timestamp, file_type)


def load_results(timestamp=None, ouputs=f"{os.getcwd()}"):
    """
    Finds SVM files created for given timestamp and transforms them to ResultList object.
    :param timestamp: string
    :return: ResultList
    """

    try:
        test_file = glob.glob("%s*%s-test*" % (ouputs+'/outputs/svm/', timestamp))[0]
        predict_file = glob.glob("%s*%s-prediction*" % (ouputs+'/outputs/svm/*', timestamp))[0]
    except IndexError:
        raise AttributeError('Given timestamp does not have files generated')

    results = ResultList()
    tagged = ResultList()
    with open(test_file, encoding="utf-8") as test, open(predict_file) as predict:
        for line, value in zip(test, predict):
            positive = "t" if line.split()[0] == '1' else "f"
            name = line.split('#')[1].strip()
            results.append(is_positive=positive, name=name, score=float(value) )

    keys = set([nm["name"].split("-")[0] for nm in results] + [nm["name"].split("-")[1] for nm in results])
    for candidate in list(keys):
        input = [ elem for elem in results if candidate == elem["name"].split("-")[0] ]
        is_positive = "t"
        if not input:
            input = [elem for elem in results if candidate == elem["name"].split("-")[1]]
            is_positive = "p"

        identified = max(input, key=lambda k: k["score"])
        identified["is_positive"] = is_positive if identified["score"] > 1.5 else "f"
        tagged.append(
            is_positive= identified["is_positive"],
            name= identified["name"],
            score= float(value)
        )
    return tagged

def svm_filename_to_timestamp(filename):
    """
    Extracts timestamp from given filename.
    :param filename: string
    :return: string
    """
    try:
        match = re.match(('%s[a-z]*-([0-9]*).*' % 'ouputs/svm/'), filename).group(1)
    except AttributeError:
        raise AttributeError('Given filename does not have correct name')
    return match

def svm_set_positions(seed_set_name):
    """
    Finds all the SVM files sets with given seed set name, transforms them to ResultList and calculates
    statistics over positive positions of these results.
    :param seed_set_name: string
    :return: None
    """
    files = glob.glob('%s%s*prediction*' % ('ouputs/svm/', seed_set_name))
    timestamps = [svm_filename_to_timestamp(filename) for filename in files]
    positions = flatten([svm_timestamp_to_results(timestamp).positive_positions() for timestamp in timestamps])
    save_stats(positions)

def svm_transform_vector(vec):
    """
    Transforms vector vec into SVM vector format:
    E.g.: 1:value_1 2:value_2 3:value_3 etc.
    :param vec: list
    :return: string
    """
    return ' '.join(["%d:%f" % (i+1, vec[i]) for i in range(len(vec))])