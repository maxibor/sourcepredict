import sys
import os
import numpy as np
import pandas as pd
from io import StringIO

from sourcepredict.sourcepredictlib import utils


def test_checks():
    assert utils.check_norm('rle') == 'RLE'
    assert utils.check_norm('None') == "no normalization"
    assert utils.check_embed('tsne') == 'TSNE'
    assert utils.check_distance('Weighted_unifrac') == 'weighted_unifrac'
    assert utils.check_weigths('Uniform') == 'uniform'
    assert utils.check_gen_seed(42) == 42
    assert type(utils.check_gen_seed(seed=None)) is int


def test_split_sinks():
    res = utils.split_sinks(StringIO(',0,1,2\n0,1,2,3\n1,4,5,6\n2,7,8,9\n'))
    assert res[0].to_string() == '   0\n0  1\n1  4\n2  7'
    assert res[1].to_string() == '   1\n0  2\n1  5\n2  8'
    assert res[2].to_string() == '   2\n0  3\n1  6\n2  9'


def test_plural():
    assert utils.plural(1) == ''
    assert utils.plural(2) == 's'


def test_get_basename():
    assert utils._get_basename('/path/to/myfile.txt') == 'myfile'


def test_class_2_dict():
    my_samp = pd.Index(['sample1', 'sample2'])
    my_classes = ['red', 'blue']
    my_pred = np.array([[0.1, 0.9], [0.2, 0.8]])

    assert utils.class2dict(samples=my_samp, classes=my_classes, pred=my_pred) == {
        'sample1': {'red': 0.1, 'blue': 0.9}, 'sample2': {'red': 0.2, 'blue': 0.8}}
