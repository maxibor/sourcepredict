#!/usr/bin/env python -W ignore::DeprecationWarning

from numpy import log, average, inf, nan, median, exp
import pandas as pd


def RLE_normalize(pd_dataframe):
    step1 = pd_dataframe.apply(log, 0)
    step2 = step1.apply(average, 1)
    step3 = step2[step2.replace([inf, -inf], nan).notnull()]
    step4_1 = step1[step1.replace(
        [inf, -inf], nan).notnull().all(axis=1)]
    step4 = step4_1.subtract(step3, 0)
    step5 = step4.apply(median, 0)
    step6 = step5.apply(exp)
    step7 = pd_dataframe.divide(step6, 1).apply(round, 1)
    return(step7)
