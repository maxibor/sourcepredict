#!/usr/bin/env python

import argparse
import os
import pandas as pd
import numpy as np


def _get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        prog='merge_kraken_res',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Merging csv count files in one table')
    parser.add_argument(
        '-o',
        dest="output",
        default=None,
        help="Output file. Default = kraken_merged.csv")

    args = parser.parse_args()

    outfile = args.output

    return(outfile)


def get_csv():
    tmp = [i for i in os.listdir() if ".csv" in i]
    return(tmp)


def _get_basename(file_name):
    if ("/") in file_name:
        basename = file_name.split("/")[-1].split(".")[0]
    else:
        basename = file_name.split(".")[0]
    return(basename)


def normalize(pd_dataframe):
    d = pd_dataframe
    step1 = d.apply(np.log, 0)
    step2 = step1.apply(np.average, 1)
    step3 = step2[step2.replace([np.inf, -np.inf], np.nan).notnull()]
    step4_1 = step1[step1.replace(
        [np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    step4 = step4_1.subtract(step3, 0)
    step5 = step4.apply(np.median, 0)
    step6 = step5.apply(np.exp)
    step7 = d.divide(step6, 1).apply(round, 1)
    return(step7)


def merge_csv(all_csv):
    df = pd.read_csv(all_csv[0], index_col=0)
    for i in range(1, len(all_csv)):
        df_tmp = pd.read_csv(all_csv[i], index_col=0)
        df = pd.merge(left=df, right=df_tmp, on='TAXID', how='outer')
    df.fillna(0, inplace=True)
    return(df)


def write_csv(pd_dataframe, outfile):
    pd_dataframe.to_csv(outfile)


if __name__ == "__main__":
    OUTFILE = _get_args()
    all_csv = get_csv()
    resdf = merge_csv(all_csv)
    resnormdf = normalize(resdf)
    write_csv(resnormdf, "kraken_merged_RLE.csv")
    write_csv(resdf, "kraken_merged_non_norm.csv")
    print(resdf)
