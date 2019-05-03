#!/usr/bin/env python3

import sys
import random
import numpy as np
import pandas as pd


def print_class(samples, classes, pred):
    """
    Print class prediction to STDOUT
    INPUT:
        samples(pandas DataFrame index): List of samples
        classes(list): list of classes
        pred(list): list of probablities predictions
    """
    print("\t----------------------")
    for i in range(0, len(samples)):
        sample = samples[i]
        print(f"\t- Sample: {sample}")
        [print(f'\t\t {i}:{round(j*100,2)}%')
         for i, j in zip(list(classes), list(pred[i, :]))]


def class2dict(samples, classes, pred):
    """
    Convert class prediction to dictionary
    INPUT:
        samples(pandas DataFrame index): List of samples
        classes(list): list of classes
        pred(list): list of probablities predictions
    OUPUT:
        resdict(dict): dictionnary of samples class probability predictions
            {samp:{class:proba}}
    """
    resdict = {}
    for i in range(0, len(samples)):
        sample = samples[i]
        resdict[sample] = {c: float(p) for (c, p) in zip(
            list(classes), list(pred[i, :]))}
    return(resdict)


def account_unk(samp_pred, umap_pred):
    """
    Account for unknown proportion
    INPUT:
        samp_pred(dict): dictionnary of samples class probability predictions
            {samp:{class:proba}}
        umap_pred(dict): dictionnary of samples unknown probability predictions
            {samp:{unknown:proba}}
    OUPUT:
        umap_pred(pandas DataFrame): sourcepredict sample proba predictions
            with unknown accounted for

    """
    for akey in umap_pred:
        umap_pred[akey] = {k: umap_pred[akey][k] *
                           (1 - samp_pred[akey]['unknown']) for k in umap_pred[akey]}
        umap_pred[akey]['unknown'] = samp_pred[akey]['unknown']
    return(pd.DataFrame(umap_pred))


def split_sinks(sink):
    """
    Split sink dataframe in individual dataframe per columns
    INPUT:
        sink(pandas DataFrame): DataFrame of sink samples
    OUTPUT:
        sinks(list) List of indivudal sink columns as df
    """
    sink_df = pd.read_csv(sink, index_col=0)
    sinks = []
    for i in sink_df.columns:
        tmp = pd.DataFrame(
            data=sink_df.loc[:, i], index=sink_df.index, columns=[i])
        sinks.append(tmp)
    return(sinks)


def check_norm(method):
    """
    Check if method is available
    INPUT:
        method(str): Normalization method
    OUTPUT:
        method(str): capitalized method
    """
    methods = ['RLE', 'CLR', 'SUBSAMPLE', 'GMPR']
    method = method.upper()
    if method not in methods:
        print("Please check the normalization method (RLE or Subsample)")
        sys.exit(1)
    else:
        return(method)


def plural(count):
    """
    Return s is count is > 1
    INPUT:
        count(int)
    OUTPUT:
        (str): '' or 's'
    """
    if count == 1:
        return('')
    else:
        return('s')


def check_embed(method):
    """
    Check if method is available
    INPUT:
        method(str): Embedding method
    OUTPUT:
        method(str): capitalized method
    """
    methods = ['TSNE', 'UMAP', 'MDS']
    method = method.upper()
    if method not in methods:
        print(f"Please check the embedding method ({' or '.join(methods)})")
        sys.exit(1)
    else:
        return(method)


def check_distance(method):
    """
    Check if method is available
    INPUT:
        method(str): distance method
    OUTPUT:
        method(str): capitalized method
    """
    methods = ['weighted_unifrac', 'unweighted_unifrac']
    method = method.lower()
    if method not in methods:
        print(f"Please check the distance method ({' or '.join(methods)})")
        sys.exit(1)
    else:
        return(method)


def check_gen_seed(seed):
    """
    Check random seed and attribute one randomly is not set
    INPUT:
        seed(int|None)
    OUTPUT:
        (int): random seed
    """
    if seed is None:
        return(random.randint(1, 10000))
    else:
        return(int(seed))


def _get_basename(file_name):
    """
    Get basename of file by splitting on "."
    INPUT:
        file_name(str): path to file
    OUTPUT
        basename(str): basename 
    """
    if ("/") in file_name:
        basename = file_name.split("/")[-1].split(".")[0]
    else:
        basename = file_name.split(".")[0]
    return(basename)


def write_out(outfile, classes, pred):
    """
    Write output file
    INPUT:
        outfile(str): path to output file
        classes()
        str_pred()
    """
    str_pred = [str(i) for i in list(pred[0])]
    with open(outfile, 'w') as f:
        f.write(",".join(list(classes))+'\n')
        f.write(",".join(list(str_pred))+'\n')
