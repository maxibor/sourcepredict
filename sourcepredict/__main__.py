#!/usr/bin/env python3

import argparse
from . sourcepredictlib import ml
from . sourcepredictlib import utils
import os
import pandas as pd
import numpy as np
import warnings
from . import __version__


def _get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        prog='SourcePredict v' + __version__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f'''
==========================================================
SourcePredict v{__version__}
Coprolite source classification
Author: Maxime Borry
Contact: <borry[at]shh.mpg.de>
Homepage & Documentation: github.com/maxibor/sourcepredict
==========================================================
        ''')
    parser.add_argument(
        'sink_table', help="path to sink TAXID count table in csv format")
    parser.add_argument(
        '-a',
        dest="alpha",
        default=0.1,
        help="Proportion of sink sample in unknown. Default = 0.1")
    parser.add_argument(
        '-s',
        dest="sources",
        default=os.path.dirname(os.path.abspath(
            __file__))+'/data/modern_gut_microbiomes_sources.csv',
        help="Path to source csv file. Default = data/modern_gut_microbiomes_sources.csv")
    parser.add_argument(
        '-l',
        dest="labels",
        default=os.path.dirname(os.path.abspath(
            __file__)) + '/data/modern_gut_microbiomes_labels.csv',
        help="Path to labels csv file. Default = data/modern_gut_microbiomes_labels.csv")
    parser.add_argument(
        '-n',
        dest="normalization",
        default='GMPR',
        help="Normalization method (RLE | Subsample | GMPR | None). Default = GMPR")
    parser.add_argument(
        '-dt',
        dest="distance",
        default='weighted_unifrac',
        help="Distance method. (unweighted_unifrac | weighted_unifrac) Default = weighted_unifrac"
    )
    parser.add_argument(
        '-r',
        dest="tax_rank",
        default='species',
        help="Taxonomic rank to use for Unifrac distances. Default = species"
    )
    parser.add_argument(
        '-me',
        dest="method",
        default='TSNE',
        help="Embedding Method. TSNE, MDS, or UMAP. Default = TSNE"
    )
    parser.add_argument(
        '-kne',
        dest="neighbors",
        default=0,
        help="Numbers of neigbors if KNN ML classication (integer or 'all'). Default = 0 (chosen by CV)"
    )
    parser.add_argument(
        '-kw',
        dest="weights",
        default='distance',
        help="Sample weight function for KNN prediction (distance | uniform). Default = distance. "
    )
    parser.add_argument(
        '-e',
        dest="embed",
        default=None,
        help="Output embedding csv file. Default = None")
    parser.add_argument(
        '-di',
        dest="dim",
        default=2,
        help="Number of dimensions to retain for dimension reduction. Default = 2"
    )
    parser.add_argument(
        '-o',
        dest="output",
        default=None,
        help="Output file basename. Default = <sample_basename>.sourcepredict.csv")
    parser.add_argument(
        '-se',
        dest="seed",
        default=42,
        help="Seed for random generator. Default = 42")
    parser.add_argument(
        '-k',
        dest="kfold",
        default=5,
        help="Number of fold for K-fold cross validation in parameter optimization. Default = 5")
    parser.add_argument(
        '-t',
        dest="threads",
        default=2,
        help="Number of threads for parallel processing. Default = 2")

    args = parser.parse_args()

    sink = args.sink_table
    alpha = float(args.alpha)
    normalization = args.normalization
    sources = args.sources
    labels = args.labels
    seed = int(args.seed)
    distance = args.distance
    rank = args.tax_rank
    method = args.method
    neighbors = str(args.neighbors)
    weights = args.weights
    dim = int(args.dim)
    output = args.output
    embed = args.embed
    kfold = int(args.kfold)
    threads = int(args.threads)

    return(sink, alpha, normalization, sources, labels, seed, distance, rank, method, neighbors, weights, dim, output, embed, kfold, threads)

def main():
    warnings.filterwarnings("ignore")
    SINK, ALPHA, NORMALIZATION, SOURCES, LABELS, SEED, DISTANCE, RANK, METHOD, NEIGHBORS, WEIGTHS, DIM, OUTPUT, EMBED_CSV, KFOLD, THREADS = _get_args()
    SEED = utils.check_gen_seed(SEED)
    np.random.seed(SEED)
    embed_method = utils.check_embed(METHOD)
    neighbors = utils.check_neighbors(NEIGHBORS)
    normalization = utils.check_norm(NORMALIZATION)
    sinks = utils.split_sinks(SINK)
    distance_method = utils.check_distance(DISTANCE)
    weigth = utils.check_weigths(WEIGTHS)
    samp_pred = {}
    print("Step 1: Checking for unknown proportion")
    if ALPHA == 0:
        print(f"\tSkipping Step 1 for alpha = {ALPHA}")
    for s in sinks:
        if ALPHA > 0 :
            sample = ''.join(list(s.columns))
            samp_pred[sample] = {}
            print(f"  == Sample: {sample} ==")
            su = ml.sourceunknown(source=SOURCES, sink=s, labels=LABELS)
            print("\tAdding unknown")
            su.add_unknown(alpha=ALPHA, seed=SEED)
            print(f"\tNormalizing ({normalization})")
            su.normalize(method=normalization, threads=THREADS)
            print("\tComputing Bray-Curtis distance")
            su.compute_distance()
            print(
                f"\tPerforming MDS embedding in {DIM} dimension{utils.plural(DIM)}")
            su.embed(n_comp=DIM, seed=SEED, out_csv=EMBED_CSV)
            print("\tKNN machine learning")
            pred = su.knn_classification(seed=SEED, threads=THREADS)
            samp_pred[sample]['unknown'] = pred[sample]['unknown']

    print("Step 2: Checking for source proportion")
    sm = ml.sourcemap(source=SOURCES, sink=SINK, labels=LABELS,
                   norm_method=normalization, threads=THREADS)
    print(f"\tComputing {distance_method} distance on {RANK} rank")
    sm.compute_distance(distance_method=distance_method, rank=RANK)
    print(f"\t{embed_method} embedding in {DIM} dimension{utils.plural(DIM)}")
    sm.embed(n_comp=DIM, method=embed_method, seed=SEED, threads=THREADS, out_csv=EMBED_CSV)
    print("\tKNN machine learning")
    source_pred = sm.knn_classification(
        kfold=KFOLD, threads=THREADS, seed=SEED, neighbors=neighbors, weigth=weigth)
    if ALPHA > 0 :
        prediction = utils.account_unk(samp_pred=samp_pred, source_pred=source_pred)
    else:
        prediction = pd.DataFrame(source_pred)
    if OUTPUT is None:
        OUTPUT = f"{utils._get_basename(SINK)}.sourcepredict.csv"
    prediction.to_csv(OUTPUT)
    print(f"Sourcepredict result written to {OUTPUT}")
    if EMBED_CSV:
        print(f"Embedding coordinates written to {EMBED_CSV}")

if __name__ == "__main__":
    main()