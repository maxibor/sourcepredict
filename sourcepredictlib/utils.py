#!/usr/bin/env python3

import sys
import random
import numpy as np
import pandas as pd


def print_class(samples, classes, pred):
    print("\t----------------------")
    for i in range(0, len(samples)):
        sample = samples[i]
        print(f"\t- Sample: {sample}")
        [print(f'\t\t {i}:{j}')
         for i, j in zip(list(classes), list(pred[i, :]))]


def class2dict(samples, classes, pred):
    resdict = {}
    for i in range(0, len(samples)):
        sample = samples[i]
        resdict[sample] = {c: float(p) for (c, p) in zip(
            list(classes), list(pred[i, :]))}
    return(resdict)


def account_unk(samp_pred, umap_pred):
    for akey in umap_pred:
        umap_pred[akey] = {k: umap_pred[akey][k] *
                           (1 - samp_pred[akey]['unknown']) for k in umap_pred[akey]}
        umap_pred[akey]['unknown'] = samp_pred[akey]['unknown']
    return(pd.DataFrame(umap_pred))


def print_ratio(classes, pred, ratio_orga):
    pred_class = {i.upper(): j for i, j in zip(
        list(classes), list(pred[0, :]))}
    num = pred_class[ratio_orga.upper()]
    denom = 0
    for i in pred_class.keys():
        if i != ratio_orga.upper():
            denom += pred_class[i]
    print(f"LogRatio {ratio_orga}/others = {np.log(num/denom)}")


def split_sinks(sink):
    sink_df = pd.read_csv(sink, index_col=0)
    sinks = []
    for i in sink_df.columns:
        tmp = pd.DataFrame(
            data=sink_df.loc[:, i], index=sink_df.index, columns=[i])
        sinks.append(tmp)
    return(sinks)


def check_norm(method):
    methods = ['RLE', 'CLR', 'SUBSAMPLE', 'GMPR']
    method = method.upper()
    if method not in methods:
        print("Please check the normalization method (RLE or Subsample)")
        sys.exit(1)
    else:
        return(method)


def check_embed(method):
    methods = ['TSNE', 'UMAP', 'MDS']
    method = method.upper()
    if method not in methods:
        print(f"Please check the embedding method ({' or '.join(methods)})")
        sys.exit(1)
    else:
        return(method)


def check_distance(method):
    methods = ['weighted_unifrac', 'unweighted_unifrac']
    method = method.lower()
    if method not in methods:
        print(f"Please check the distance method ({' or '.join(methods)})")
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
