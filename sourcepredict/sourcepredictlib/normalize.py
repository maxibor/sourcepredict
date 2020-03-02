#!/usr/bin/env python3

import numpy as np
import pandas as pd
from functools import partial
import multiprocessing


def RLE_normalize(pd_dataframe):
    """Normalize with Relative Log Expression

    Args:
        pd_dataframe (pandas DataFrame): TAXID count dataframe,
            colums as Samples, Rows as TAXIDs
    Returns:
        pandas DataFrame: RLE Normalized datafrane. Colums as Samples, Rows as TAXIDs
    Example:
        >>> RLE_normalize(pd.DataFrame)
    """

    step1 = pd_dataframe.apply(np.log, 0)
    step2 = step1.apply(np.average, 1)
    step3 = step2[step2.replace([np.inf, -np.inf], np.nan).notnull()]
    step4_1 = step1[step1.replace(
        [np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    step4 = step4_1.subtract(step3, 0)
    step5 = step4.apply(np.median, 0)
    step6 = step5.apply(np.exp)
    step7 = pd_dataframe.divide(step6, 1).apply(round, 1)
    return(step7.dropna(axis=1))


def subsample_normalize_pd(pd_dataframe):
    """Normalize with Subsampling

    Args:
        pd_dataframe (pandas DataFrame): TAXID count dataframe,
            colums as Samples, Rows as TAXIDs
    Returns:
       pandas DataFrame: Subsample Normalized dataframe. Colums as Samples, Rows as TAXIDs
    """

    def subsample_normalize(serie, omax):
        """Subsample normalization column wise

        imin: minimum of input range
        imax: maximum of input range
        omin: minimum of output range
        omax: maximum of output range
        x in [imin, imax]
        f(x) in [omin, omax]

                 x - imin
        f(x) = ------------ x(omax - omin) + omin
               imax - imin


        Args:
            serie (pandas Series): Indivudal Sample Column
            omax (int): maximum of output range
        Returns:
            pandas Series: normalized pandas Series
        """

        imin = min(serie)
        imax = max(serie)
        omin = 0
        if imax > 0:
            newserie = serie.apply(lambda x: (
                (x - imin)/(imax - imin)*(omax-omin)+omin))
        else:
            newserie = serie
        return(newserie)

    step1 = pd_dataframe.apply(max, 1)
    themax = max(step1)

    step2 = pd_dataframe.apply(
        subsample_normalize, axis=0, args=(themax,))
    step3 = step2.apply(np.floor, axis=1)
    return(step3.dropna(axis=1))


def gmpr_size_factor(col, ar):
    """Generate GMPR size factor

    Args:
        col (int): columm index of the numpy array
        ar (numpy array): numpy array of TAXID counts,
            colums as Samples, Rows as TAXIDs
    Returns:
        float: GMPR size factor per column
    """
    pr = np.apply_along_axis(lambda x: np.divide(ar[:, col], x), 0, ar)
    pr[np.isinf(pr)] = np.nan
    pr[pr == 0] = np.nan
    pr_median = np.nanmedian(pr, axis=0)
    return(np.exp(np.mean(np.log(pr_median))))


def GMPR_normalize(df, process):
    """Compute GMPR normalization

    Global Mean of Pairwise Ratios
    Chen, L., Reeve, J., Zhang, L., Huang, S., Wang, X., & Chen, J. (2018). 
    GMPR: A robust normalization method for zero-inflated count data 
    with application to microbiome sequencing data. 
    PeerJ, 6, e4600.

    Args:
        df (pandas Dataframe): TAXID count dataframe,
            colums as Samples, Rows as TAXIDs
        process (int): number of process for parallelization
    """
    ar = np.asarray(df)

    gmpr_sf_partial = partial(gmpr_size_factor, ar=ar)
    with multiprocessing.Pool(process) as p:
        sf = p.map(gmpr_sf_partial, list(range(np.shape(ar)[1])))

    return(pd.DataFrame(np.divide(ar, sf), index=df.index, columns=df.columns).dropna(axis=1))
