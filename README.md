[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict) [![Anaconda-Server Badge](https://anaconda.org/maxibor/sourcepredict/badges/installer/conda.svg)](https://conda.anaconda.org/maxibor)

# SourcePredict

Prediction/source tracking of sample source using a random forest approach

## Installation

```
$ conda install -c maxibor sourcepredict
```

## Example

```bash
$ sourcepredict -r canis_familiaris ./data/test/dog_test_sample.csv
Training classifier on 2 cores...
Training Accuracy: 1.0
=================
Canis_familiaris:0.96
Homo_sapiens:0.003
Sus_scrofa:0.0
unknown:0.037
LogRatio canis_familiaris/others = 3.1780538303479458
```

## Help

```
$ sourcepredict -h
usage: SourcePredict v0.1 [-h] [-a ALPHA] [-s SOURCES] [-l LABELS] [-r RATIO]
                          [-n NORMALIZATION] [-o OUTPUT] [-se SEED]
                          [-t THREADS]
                          otu_table

==========================================================
SourcePredict v0.1
Coprolite source classification
Author: Maxime Borry
Contact: <borry[at]shh.mpg.de>
Homepage & Documentation: github.com/maxibor/sourcepredict
==========================================================


positional arguments:
  otu_table         path to otu table in csv format

optional arguments:
  -h, --help        show this help message and exit
  -a ALPHA          Proportion of sink sample in unknown. Default = 0.1
  -s SOURCES        Path to source csv file. Default =
                    ./data/sourcepredict_sources.csv
  -l LABELS         Path to labels csv file. Default =
                    ./data/sourcepredict_labels.csv
  -r RATIO          Target organism for ratio calculation. Default =
                    'Homo_sapiens'
  -n NORMALIZATION  Normalization method (RLE | CLR | Subsample). Default =
                    RLE
  -o OUTPUT         Output file basename. Default =
                    <sample_basename>.sourcepredict.csv*
  -se SEED          Seed for random generator. Default = None (randomly
                    generated)
  -t THREADS        Number of threads for parallel processing. Default = 2
```

## Sourcepredict source file

- The sources were obtained with the [Kraken based pipeline](utils/kraken_pipeline/kraken_pipe.nf) included in this repository, using the [*Dustmasked MiniKraken DB 4GB*](https://ccb.jhu.edu/software/kraken/dl/minikraken_20171101_4GB_dustmasked.tgz).  
- The source file if here [data/dog_human_pig_sources.csv](data/dog_human_pig_sources.csv)
- The labels for this source file is here [data/labels.csv](data/labels.csv)


### Current species included in the source file

- *Sus scrofa*
- *Homo sapiens*
- *Canis familiaris*

### Updating the source file 

To update the sourcefile with new kraken results, see the instruction in the [dedicated Jupyter notebook](notebooks/merge_new_data.ipynb) 
