import sys
import os
import numpy as np
import pandas as pd

parentScriptDir = "/".join(os.path.dirname(
    os.path.realpath(__file__)).split("/")[:-1])
sys.path.append(parentScriptDir+"/sourcepredictlib")

import normalize


def test_RLE():
    """
    Test RLE normalization
    """

    input_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output_df = pd.DataFrame(
        [[1.0, 2.0, 2.0], [5.0, 5.0, 5.0], [9.0, 8.0, 7.0]])

    assert normalize.RLE_normalize(
        input_df).all().all() == output_df.all().all()


def test_subsample():
    """
    Test subsample normalization
    """

    input_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output_df = pd.DataFrame(
        [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [9.0, 9.0, 9.0]])

    assert normalize.subsample_normalize_pd(
        input_df).all().all() == output_df.all().all()


def test_gmpr_size_factor():
    """
    Test GMPR normalization size factor
    """

    input_ar = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output = 1.0137003325955667
    assert normalize.gmpr_size_factor(col=1, ar=input_ar) == output


def test_GMPR():
    """
    Test GMPR normalization
    """

    input_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output_df = pd.DataFrame([[1.2331060371652351, 1.9729696594643762, 2.4662120743304703],
                              [4.932424148660941, 4.932424148660941,
                                  4.932424148660941],
                              [8.631742260156646, 7.891878637857505, 7.398636222991411]])
    assert normalize.GMPR_normalize(
        input_df, 1).all().all() == output_df.all().all()
    assert normalize.GMPR_normalize(
        input_df, 2).all().all() == output_df.all().all()
