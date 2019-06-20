import sys
import os
import pandas as pd
import random

parentScriptDir = "/".join(os.path.dirname(
    os.path.realpath(__file__)).split("/")[:-1])
sys.path.append(parentScriptDir+"/sourcepredictlib")
random.seed(42)

import ml
import utils


def test_sourceunknown_init():

    PYTHONHASHSEED = 0

    labels = os.path.dirname(os.path.abspath(
        __file__)) + '/../data/modern_gut_microbiomes_labels.csv'
    sources = os.path.dirname(os.path.abspath(
        __file__))+'/../data/modern_gut_microbiomes_sources.csv'
    sink_file = os.path.dirname(os.path.abspath(
        __file__))+'/../data/test/dog_test_sample.csv'
    sink = utils.split_sinks(sink_file)[0]

    su = ml.sourceunknown(source=sources, sink=sink, labels=labels)

    assert su.ref.shape == (5664, 432)
    assert su.y.shape == (432,)
    assert su.y_unk.shape == (432,)
    assert su.tmp_sink.shape == (570, 1)
    assert su.combined.shape == (5664, 433)
    assert hash(str(su.combined)) == -1867655657877779130


# def test_sourceunknown_add_unkown():
#     PYTHONHASHSEED = 0
#     labels = os.path.dirname(os.path.abspath(
#         __file__)) + '/../data/modern_gut_microbiomes_labels.csv'
#     sources = os.path.dirname(os.path.abspath(
#         __file__))+'/../data/modern_gut_microbiomes_sources.csv'
#     sink_file = os.path.dirname(os.path.abspath(
#         __file__))+'/../data/test/dog_test_sample.csv'
#     sink = utils.split_sinks(sink_file)[0]

#     su = ml.sourceunknown(source=sources, sink=sink, labels=labels)
#     su.add_unknown(alpha=0.1, seed=42)

#     assert su.ref_u.shape == (570, 144)
#     assert su.ref_u.dtypes ==
#     assert hash(str(su.ref_u.columns)) == 5867343156924504419
#     assert hash(str(su.ref_u.index)) == 4313932402357376923
#     assert hash(str(su.ref_u_labs)) == -5906979299339891562


# def test_sourceunknown_normalized():

#     labels = os.path.dirname(os.path.abspath(
#         __file__)) + '/data/modern_gut_microbiomes_labels.csv'
#     sources = os.path.dirname(os.path.abspath(
#         __file__))+'/data/modern_gut_microbiomes_sources.csv'
#     sink_file = os.path.dirname(os.path.abspath(
#         __file__))+'data/test/dog_test_sample.csv'
#     sink = utils.split_sinks(sink_file)[0]

#     su = ml.sourceunknown(source=sources, sink=sink, labels=labels)
#     su.add_unknown(alpha=0.1, seed=42)
#     su_rle = su.normalize(method='rle', threads=1)
#     su_subsample = su.normalize(method='subsample', threads=1)
#     su_gmpr = su.normalize(method='gmpr', threads=1)
