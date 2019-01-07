#!/usr/bin/env python3

import sys
import random
import numpy as np


def print_class(classes, pred):
    [print(f'{i}:{j}') for i, j in zip(list(classes), list(pred[0, :]))]


def print_ratio(classes, pred, ratio_orga):
    pred_class = {i.upper(): j for i, j in zip(
        list(classes), list(pred[0, :]))}
    num = pred_class[ratio_orga.upper()]
    denom = 0
    for i in pred_class.keys():
        if i != ratio_orga.upper():
            denom += pred_class[i]
    print(f"LogRatio {ratio_orga}/others = {np.log(num/denom)}")


def check_norm(method):
    methods = ['RLE', 'CLR', 'SUBSAMPLE']
    method = method.upper()
    if method not in methods:
        print("Please check the normalization method (RLE or Subsample)")
        sys.exit(1)
    else:
        return(method)


def check_gen_seed(seed):
    if seed is None:
        return(random.randint(1, 10000))
    else:
        return(int(seed))


def _get_basename(file_name):
    if ("/") in file_name:
        basename = file_name.split("/")[-1].split(".")[0]
    else:
        basename = file_name.split(".")[0]
    return(basename)


def write_out(outfile, classes, pred):
    str_pred = [str(i) for i in list(pred[0])]
    with open(outfile, 'w') as f:
        f.write(",".join(list(classes))+'\n')
        f.write(",".join(list(str_pred))+'\n')
