#!/usr/bin/env python3

import numpy as np
import pandas as pd
from functools import partial
import multiprocessing


def RLE_normalize(pd_dataframe):
    '''
    Normalize with Relative Log Expression
    INPUT:
        pd_dataframe(pandas DataFrame): Colums as Samples, Rows as OTUs
    OUTPUT:
        step7(pandas DataFrame): RLE Normalized. Colums as Samples, Rows as OTUs
    '''
    step1 = pd_dataframe.apply(np.log, 0)
    step2 = step1.apply(np.average, 1)
    step3 = step2[step2.replace([np.inf, -np.inf], np.nan).notnull()]
    step4_1 = step1[step1.replace(
        [np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    step4 = step4_1.subtract(step3, 0)
    step5 = step4.apply(np.median, 0)
    step6 = step5.apply(np.exp)
    step7 = pd_dataframe.divide(step6, 1).apply(round, 1)
    return(step7)


def subsample_normalize_pd(pd_dataframe):
    '''
    Normalize with Subsampling
    INPUT:
        pd_dataframe(pandas DataFrame): Colums as Samples, Rows as OTUs
    OUTPUT:
        step7(pandas DataFrame): SubSample Normalized. Colums as Samples, Rows as OTUs
    '''
    def subsample_normalize(serie, omax):
        '''
        imin: minimum of input range
        imax: maximum of input range
        omin: minimum of output range
        omax: maximum of output range
        x in [imin,imax]
        f(x) in [omin, omax]

                 x - imin
        f(x) = ------------ x (omax - omin) + omin
               imax - imin

        '''
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
    return(step3)


def CLR_normalize(pd_dataframe):
    d = pd_dataframe
    d = d+1
    step1_1 = d.apply(np.log, 0)
    step1_2 = step1_1.apply(np.average, 0)
    step1_3 = step1_2.apply(np.exp)
    step2 = d.divide(step1_3, 1)
    step3 = step2.apply(np.log, 0)
    return(step3)


def gmpr_size_factor(col, ar):
    pr = np.apply_along_axis(lambda x: np.divide(ar[:, col], x), 0, ar)
    pr[np.isinf(pr)] = np.nan
    pr[pr == 0] = np.nan
    pr_median = np.nanmedian(pr, axis=0)
    return(np.exp(np.mean(np.log(pr_median))))


def GMPR_normalize(df, process):
    """
    Global Mean of Pairwise Ratios
    Chen, L., Reeve, J., Zhang, L., Huang, S., Wang, X., & Chen, J. (2018). 
    GMPR: A robust normalization method for zero-inflated count data 
    with application to microbiome sequencing data. 
    PeerJ, 6, e4600.
    """
    ar = np.asarray(df)

    gmpr_sf_partial = partial(gmpr_size_factor, ar=ar)
    with multiprocessing.Pool(process) as p:
        sf = p.map(gmpr_sf_partial, list(range(np.shape(ar)[1])))

    return(pd.DataFrame(np.divide(ar, sf), index=df.index, columns=df.columns))
