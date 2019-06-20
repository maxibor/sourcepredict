import sys
import os

parentScriptDir = "/".join(os.path.dirname(
    os.path.realpath(__file__)).split("/")[:-1])
sys.path.append(parentScriptDir+"/sourcepredictlib")

import utils


def test_checks():
    assert utils.check_norm('rle') == 'RLE'
    assert utils.check_embed('tsne') == 'TSNE'
    assert utils.check_distance('Weighted_unifrac') == 'weighted_unifrac'
    assert utils.check_gen_seed(42) == 42
    assert type(utils.check_gen_seed(seed=None)) is int
