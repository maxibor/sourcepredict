import sys
import os
import pytest
import hashlib

parentScriptDir = "/".join(os.path.dirname(
    os.path.realpath(__file__)).split("/")[:-1])
sys.path.append(parentScriptDir+"/sourcepredictlib")

import ml
import utils


def generate_pd_md5(pd_obj):
    """
    Generates MD5 hash of Pandas object (Series or DataFrame)
    """
    return(hashlib.md5(pd_obj.to_string().encode()).hexdigest())


def generate_str_md5(obj):
    """
    Generates MD5 hash from object casted to string
    """
    return(hashlib.md5(str(obj).encode()).hexdigest())


@pytest.fixture(autouse=True)
def su():
    """
    Initialized SourceUnknown object
    """
    labels = os.path.dirname(os.path.abspath(
        __file__)) + '/../data/test/training/test_labels.csv'
    sources = os.path.dirname(os.path.abspath(
        __file__))+'/../data/test/training/test_sources.csv'
    sink_file = os.path.dirname(os.path.abspath(
        __file__))+'/../data/test/testing/test_data.csv'
    sink = utils.split_sinks(sink_file)[0]

    return(ml.sourceunknown(source=sources, sink=sink, labels=labels))


def test_init_sourceunknown(su):
    assert generate_pd_md5(su.ref) == '166faef750858398d21292330532fc36'
    assert generate_pd_md5(su.y) == 'df0003c57f029ba8df08f2e794a6f29c'
    assert generate_pd_md5(su.y_unk) == '3512f5391306f31bc111e45a2e3262e5'
    assert generate_pd_md5(su.tmp_sink) == 'b8b76385619f3e919b292fdc5d54d330'
    assert generate_pd_md5(su.combined) == '70c6bd781f5458a04c882fc37fc9fae5'


def test_sourceunknown_add_unkown(su):
    su.add_unknown(alpha=0.1, seed=42)
    assert su.ref_u.shape == (5, 60)
    assert su.ref_u.dtypes.value_counts().to_string() == 'float64    60'
    assert generate_str_md5(
        su.ref_u.columns) == 'f916bcfc1c94e1f1ca495ff6ab608dbd'
    assert generate_str_md5(
        su.ref_u.index) == '0dc389f3883bc3b985293499b3e8b923'
    assert generate_str_md5(
        su.ref_u_labs) == '713efce769c9a304537d3ef0586d6548'


def test_sourceunknown_normalized(su):
    su.add_unknown(alpha=0.1, seed=42)
    su.normalize(method='GMPR', threads=1)
    assert su.normalized_ref_u.shape == (5, 181)
    assert generate_pd_md5(su.labels) == 'b3d3be3f23348e803f61329820fbd49b'
    assert generate_pd_md5(su.sink) == 'fb6d76db32a82bff224b94bd276c3d7c'
    assert generate_pd_md5(su.y_unk) == '5fc8eb596caa8776195c82930a7656f2'


def test_compute_distance(su):
    su.add_unknown(alpha=0.1, seed=42)
    su.normalize(method='GMPR', threads=2)
    su.compute_distance(rank='species')

    assert su.bc.shape == (181, 181)
    assert su.normalized_rank.shape == (181, 2)


def test_embed(su):
    su.add_unknown(alpha=0.1, seed=42)
    su.normalize(method='GMPR', threads=2)
    su.compute_distance(rank='species')
    su.embed(n_comp=2, seed=42, out_csv=None)

    assert su.my_embed.shape == (181, 2)
    assert generate_str_md5(
        su.my_embed.columns) == '2085680ed982ddbafe984eafee3524c5'
    assert generate_str_md5(
        su.my_embed.index) == '2c80ec071249282eb41526282acd3b03'
    assert su.ref_u.shape == (180, 3)
    assert su.sink.shape == (1, 2)


def test_unk_ml(su):
    su.add_unknown(alpha=0.1, seed=42)
    su.normalize(method='GMPR', threads=2)
    su.compute_distance(rank='species')
    su.embed(n_comp=2, seed=42, out_csv=None)
    res = su.ml(seed=42, threads=2)
    assert round(res['metagenomebis']['known'], 3) == 0.952
    assert round(res['metagenomebis']['unknown'], 3) == 0.048


@pytest.fixture(autouse=True)
def sm():
    """
    Initialized SourceUnknown object
    """
    labels = os.path.dirname(os.path.abspath(
        __file__)) + '/../data/test/training/test_labels.csv'
    sources = os.path.dirname(os.path.abspath(
        __file__))+'/../data/test/training/test_sources.csv'
    sink_file = os.path.dirname(os.path.abspath(
        __file__))+'/../data/test/testing/test_data.csv'

    return(ml.sourcemap(source=sources, sink=sink_file, labels=labels,
                        norm_method='GMPR', threads=1))


def test_sourcemap_init(sm):
    assert generate_pd_md5(sm.normalized) == 'a6cbd65eac3135ffee5d6ca7e7d0aa62'


def test_sourcemap_dist(sm):
    sm.compute_distance(distance_method='weighted_unifrac', rank='species')
    assert generate_pd_md5(sm.wu) == 'fe2470a7842811288d226a0e14da8344'


def test_sourcemap_embed_TSNE(sm):
    sm.compute_distance(distance_method='weighted_unifrac', rank='species')
    sm.embed(n_comp=2, method='TSNE', seed=42, out_csv=None)

    assert generate_pd_md5(sm.my_embed) == '63dcc7ff98d71c860fcd44791c28cecb'
    assert generate_pd_md5(sm.ref_t) == 'b94392f2a8cbeac536fa3222d41e2477'
    assert generate_pd_md5(sm.sink_t) == 'e63b98a71ea1c444bd97659ba8a322f8'


def test_sourcemap_embed_MDS(sm):
    sm.compute_distance(distance_method='weighted_unifrac', rank='species')
    sm.embed(n_comp=2, method='MDS', seed=42, out_csv=None)

    assert generate_pd_md5(sm.my_embed) in [
        'a05d18d0d35a39ec3cd63464089fb2bf', 'f876bd2095f36beb3ec8925558148a15']
    assert generate_pd_md5(sm.ref_t) == '96eb677be04de2db784048428b1b581d'
    assert generate_pd_md5(sm.sink_t) == '9f7a4b87e58194db1cd7357a031de217'


def test_sourcemap_ml(sm):
    sm.compute_distance(distance_method='weighted_unifrac', rank='species')
    sm.embed(n_comp=2, method='MDS', seed=42, out_csv=None)
    res = sm.knn_classification(kfold=3, threads=1, seed=42)

    assert round(res['metagenomebis']['Bacillus_subtilis'], 3) == 0.875
    assert round(res['metagenomebis']['Escherichia_coli'], 3) == 0.125
    assert round(res['metagenome']['Bacillus_subtilis'], 3) == 0.875
    assert round(res['metagenome']['Escherichia_coli'], 3) == 0.125
