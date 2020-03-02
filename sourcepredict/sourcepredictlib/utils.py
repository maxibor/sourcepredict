#!/usr/bin/env python3

import sys
import random
import pandas as pd


def print_class(samples, classes, pred):
    """Print class prediction to STDOUT

    Args:
        samples(pandas DataFrame index): List of samples
        classes(list): list of classes
        pred(list): list of probablities predictions
    """

    print("\t----------------------")
    for ix, sample in enumerate(samples):
        print(f"\t- Sample: {sample}")
        [print(f'\t\t {i}:{round(j*100,2)}%') for i, j 
        in zip(list(classes), list(pred[ix, :]))]


def class2dict(samples, classes, pred):
    """Convert class prediction to dictionary

    Args:
        samples(pandas DataFrame index): List of samples
        classes(list): list of classes
        pred(list): list of probablities predictions
    Returns:
        dict: dictionnary of samples class probability predictions
            {samp: {class: proba}}
    """

    resdict = {}
    for ix, sample in enumerate(samples):
        resdict[sample] = {c: float(p) for (c, p) in zip(
            list(classes), list(pred[ix, :]))}
    return(resdict)


def account_unk(samp_pred, source_pred):
    """Account for unknown proportion

    Args:
        source_pred(dict): dictionnary of samples class probability predictions
            {samp: {class: proba}}
        samp_pred(dict): dictionnary of samples unknown probability predictions
            {samp: {unknown: proba}}
    Returns:
        pandas DataFrame: sourcepredict sample proba predictions
            with unknown accounted for
    """

    for akey in source_pred:
        source_pred[akey] = {k: source_pred[akey][k] *
                           (1 - samp_pred[akey]['unknown']) for k in source_pred[akey]}
        source_pred[akey]['unknown'] = samp_pred[akey]['unknown']
    return(pd.DataFrame(source_pred))


def account_unk_gmm(samp_pred):
    """Account for unknown proportion after one step GMM source prediction

    Args:
        samp_pred(dict): dictionnary of samples class probability predictions
            {samp: {class: proba}}
    Returns:
        pandas DataFrame: sourcepredict sample proba predictions
            with unknown accounted for
    """
    for sample in samp_pred:
        smp_tot = 0
        for source in samp_pred[sample]:
            smp_tot += samp_pred[sample][source]
        samp_pred[sample]['unknown'] = 1 - smp_tot
    return(samp_pred)


def split_sinks(sink):
    """Split sink dataframe in individual dataframe per columns

    Args:
       sink(pandas DataFrame): DataFrame of sink samples
    Returns:
        list of pandas Dataframes: List of indivudal sink columns as pandas Dataframes
    """

    sink_df = pd.read_csv(sink, index_col=0)
    sinks = []
    for i in sink_df.columns:
        tmp = pd.DataFrame(
            data=sink_df.loc[:, i], index=sink_df.index, columns=[i])
        sinks.append(tmp)
    return(sinks)


def check_norm(method):
    """Check if normalization method is valid

    Args:
        method(str): Normalization method
    Returns:
        str: capitalized normalization method name
    Example:
        >>> check_norm('rle')
        'RLE'
    """
    methods = ['RLE', 'SUBSAMPLE', 'GMPR']
    if method == 'None':
        return("no normalization")
    method = method.upper()
    if method not in methods:
        print("Please check the normalization method (GMPR, RLE, or Subsample)")
        sys.exit(1)
    else:
        return(method)


def check_neighbors(neighbors):
    """Check if neighbors is valid

    Args:
        neighbors(str): number of neighbors or 'all'
    Returns:
        numbers of neighbors(int) or 'all'(str) 
    Example:
        >>> check_neighbors('12')
        12
        >>> check_neighbors('all')
        'all'
    """
    if neighbors.isdigit():
        return(int(neighbors))
    elif neighbors.lower() == 'all':
        return('all')
    else:
        print("Please check the neighbors arguments (interger or 'all')")
        sys.exit(1)


def check_embed(method):
    """Check if embedding method is valid

    Args:
        method(str): Embedding method
    Returns:
        str: capitalized embedding method name
    Example:
        >>> check_embed('tsne')
        'TSNE'
    """

    methods = ['TSNE', 'UMAP', 'MDS']
    method = method.upper()
    if method not in methods:
        print(f"Please check the embedding method ({' or '.join(methods)})")
        sys.exit(1)
    else:
        return(method)


def check_distance(method):
    """Check if distance method is valid

    Args:
        method(str): distance method
    Returns:
        str: capitalized distance method name
    Example:
        >>> check_distance('Weighted_unifrac')
        'weighted_unifrac'
    """

    methods = ['weighted_unifrac', 'unweighted_unifrac']
    method = method.lower()
    if method not in methods:
        print(f"Please check the distance method ({' or '.join(methods)})")
        sys.exit(1)
    else:
        return(method)


def check_weigths(weight):
    """Check if weight parameter for KNN is valid

    Args:
        method(str): weight parameter
    Returns:
        str: lowercase weigth parameter
    Example:
        >>> check_weigths('Uniform')
        'uniform'
    """
    weights = ['uniform', 'distance']
    weight = weight.lower()
    if weight not in weights:
        print(f"Please check the weight parameter ({' or '.join(weights)})")
        sys.exit(1)
    else:
        return(weight)

def check_gen_seed(seed, amin=1, amax=10000):
    """Check random seed

    Args:
        seed(int or None): random seed
        amin (int, optional): Lower boundary for random sampling of seed. 
            Defaults to 1.
        amax (int, optional): Upper boundary for random sampling of seed. 
            Defaults to 10000.
    Returns:
        int: random seed sampled between 1 and 10000
    Example:
        >>> check_gen_seed(42)
        42
    """

    if seed is None:
        return(random.randint(1, 10000))
    else:
        return(int(seed))


def plural(count):
    """Return s is count is > 1

    Args:
        count(int): number of occurences
    Returns:
        str: '' or 's'
    Example:
        >>> plural(3)
        's'
    """

    if count == 1:
        return('')
    else:
        return('s')


def _get_basename(file_name):
    """Get file basename

    Get basename of file by splitting on "."

    Args:
        file_name(str): path to file
    Returns:
        str: file basename
    Example:
        >>> _get_basename('/path/to/myfile.txt')
        'myfile'
    """

    if ("/") in file_name:
        basename = file_name.split("/")[-1].split(".")[0]
    else:
        basename = file_name.split(".")[0]
    return(basename)
