#!/usr/bin/env python3

import argparse
from sourcepredict.lib.ml import sourceunknown
from sourcepredict.lib.ml import sourcemap
from sourcepredict.lib import utils
import os
import pandas as pd
import numpy as np
import warnings
import sys

__version__ = '0.3.3'


def _get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        prog='SourcePredict v' + str(__version__),
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
    parser.add_argument('otu_table', help="path to otu table in csv format")
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
        help="Normalization method (RLE | CLR | Subsample | GMPR). Default = GMPR")
    parser.add_argument(
        '-dt',
        dest="distance",
        default='weighted_unifrac',
        help="Distance method. (unweighted_unifrac | weighted_unifrac) Default = weighted_unifrac"
    )
    parser.add_argument(
        '-me',
        dest="method",
        default='TSNE',
        help="Embedding Method. TSNE or UMAP. Default = TSNE"
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
        help="Number of fold for K-fold cross validation in feature selection and parameter optimization. Default = 5")
    parser.add_argument(
        '-t',
        dest="threads",
        default=2,
        help="Number of threads for parallel processing. Default = 2")

    args = parser.parse_args()

    sink = args.otu_table
    alpha = float(args.alpha)
    normalization = args.normalization
    sources = args.sources
    labels = args.labels
    seed = int(args.seed)
    distance = args.distance
    method = args.method
    dim = int(args.dim)
    output = args.output
    embed = args.embed
    kfold = int(args.kfold)
    threads = int(args.threads)

    return(sink, alpha, normalization, sources, labels, seed, distance, method, dim, output, embed, kfold, threads)


def main():
    SINK, ALPHA, NORMALIZATION, SOURCES, LABELS, SEED, DISTANCE, METHOD, DIM, OUTPUT, EMBED_CSV, KFOLD, THREADS = _get_args()
    SEED = utils.check_gen_seed(SEED)
    np.random.seed(SEED)
    embed_method = utils.check_embed(METHOD)
    normalization = utils.check_norm(NORMALIZATION)
    sinks = utils.split_sinks(SINK)
    predictions = {}
    distance_method = utils.check_distance(DISTANCE)
    tax_rank = "species"
    samp_pred = {}
    print("Step 1: Checking for unknown proportion")
    for s in sinks:
        sample = ''.join(list(s.columns))
        samp_pred[sample] = {}
        print(f"  == Sample: {sample} ==")
        a = sourceunknown(source=SOURCES, sink=s, labels=LABELS)
        print("\tAdding unknown")
        a.add_unknown(alpha=ALPHA, seed=SEED)
        print(f"\tNormalizing ({normalization})")
        a.normalize(method=normalization, threads=THREADS)
        print("\tComputing Bray-Curtis distance")
        a.compute_distance(rank=tax_rank)
        print(
            f"\tPerforming MDS embedding in {DIM} dimension{utils.plural(DIM)}")
        a.embed(n_comp=DIM, seed=SEED, out_csv=EMBED_CSV)

        print("\tKNN machine learning")
        pred = a.ml(seed=SEED, threads=THREADS)
        samp_pred[sample]['unknown'] = pred[sample]['unknown']

    print("Step 2: Checking for source proportion")
    u = sourcemap(source=SOURCES, sink=SINK, labels=LABELS,
                  norm_method=NORMALIZATION, threads=THREADS)
    print(f"\tComputing {distance_method} distance on {tax_rank} rank")
    u.compute_distance(distance_method=distance_method, rank=tax_rank)
    print(f"\t{embed_method} embedding in {DIM} dimension{utils.plural(DIM)}")
    u.embed(n_comp=DIM, method=embed_method, seed=SEED, out_csv=EMBED_CSV)
    print("\tKNN machine learning")
    umap_pred = u.knn_classification(
        kfold=KFOLD, threads=THREADS, seed=SEED)

    prediction = utils.account_unk(samp_pred=samp_pred, umap_pred=umap_pred)
    if OUTPUT is None:
        OUTPUT = f"{utils._get_basename(SINK)}.sourcepredict.csv"
    prediction.to_csv(OUTPUT)
    print(f"Sourcepredict result written to {OUTPUT}")
    if EMBED_CSV:
        print(f"Embedding coordinates written to {EMBED_CSV}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
