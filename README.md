[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict) [![Anaconda-Server Badge](https://anaconda.org/maxibor/sourcepredict/badges/installer/conda.svg)](https://conda.anaconda.org/maxibor)

<img src="img/sourcepredict_logo.png" width="300">

Prediction/source tracking of sample source using a random forest approach

## Installation

```
$ conda install -c maxibor sourcepredict
```

## Example

```bash
$ sourcepredict -r canis_familiaris ./data/test/dog_test_sample.csv
Performing 3 fold cross validation on 2 cores...
Training classifier with best parameters on 2 cores...
Training Accuracy: 1.0
=================
Canis_familiaris:0.9058876469899879
Homo_sapiens:0.00920415762430391
Sus_scrofa:0.0003424958927924832
UNKNOWN:0.08456569949291566
LogRatio canis_familiaris/others = 2.2644259750872995
```

## Help

```
$ sourcepredict -h
usage: SourcePredict v0.1.1 [-h] [-a ALPHA] [-s SOURCES] [-l LABELS]
                            [-r RATIO] [-n NORMALIZATION] [-o OUTPUT]
                            [-se SEED] [-k KFOLD] [-t THREADS]
                            otu_table

==========================================================
SourcePredict v0.1.1
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
                    data/sourcepredict_sources.csv
  -l LABELS         Path to labels csv file. Default =
                    data/sourcepredict_labels.csv
  -r RATIO          Target organism for ratio calculation. Default =
                    'Homo_sapiens'
  -n NORMALIZATION  Normalization method (RLE | CLR | Subsample). Default =
                    RLE
  -o OUTPUT         Output file basename. Default =
                    <sample_basename>.sourcepredict.csv
  -se SEED          Seed for random generator. Default = None (randomly
                    generated)
  -k KFOLD          Number of fold for K-fold cross validation in feature
                    selection and parameter optimization. Default = 3
  -t THREADS        Number of threads for parallel processing. Default = 2
```

## Sourcepredict source file

- The sources were obtained with the [Kraken based pipeline](utils/kraken_pipeline/kraken_pipe.nf) included in this repository, using the [*Dustmasked MiniKraken DB 4GB*](https://ccb.jhu.edu/software/kraken/dl/minikraken_20171101_4GB_dustmasked.tgz).  
- The source file is here [data/sourcepredict_sources.csv](data/sourcepredict_sources.csv)
- The label file for this source file is here [data/sourcepredict_labels.csv](data/sourcepredict_labels.csv)


### Current species included in the source file

- *Sus scrofa*
- *Homo sapiens*
- *Canis familiaris*

### Updating the source file 

To update the sourcefile with new kraken results, see the instruction in the [dedicated Jupyter notebook](notebooks/merge_new_data.ipynb) 
