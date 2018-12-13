#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import pandas as pd


def RLE_normalize(pd_dataframe):
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
