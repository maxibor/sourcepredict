[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict) [![Anaconda-Server Badge](https://anaconda.org/maxibor/sourcepredict/badges/installer/conda.svg)](https://conda.anaconda.org/maxibor)

<img src="img/sourcepredict_logo.png" width="300">

Prediction/source tracking of sample source using a random forest approach

## Installation

```
$ conda install -c etetoolkit -c bioconda -c maxibor sourcepredict
```

## Example

```bash
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/test/dog_test_sample.csv?token=AIOyNX-Styi0FWlY-9ZILyGbh8EpEYmDks5cd_k4wA%3D%3D -O dog_test_sample.csv
$ sourcepredict dog_test_sample.csv
== Sample: ERR1915662 ==
Step 1: Checking for unknown proportion
	Adding unknown
	Normalizing
	Feature engineering
	Random forest machine learning
	Performing 3 fold cross validation on 2 cores...
	Training random forest classifier with best parameters on 2 cores...
	-> Testing Accuracy: 1.0
	----------------------
	- Unknown: 0.0%
Step 2: Checking for source proportion
	Computing weighted unifrac distance on species rank
	UMAP embedding
	KNN machine learning
	Performing 3 fold cross validation on 2 cores...
	-> Testing Accuracy: 1.0
	----------------------
	- Canis_familiaris:1.0
	- Homo_sapiens:0.0
	- Sus_scrofa:0.0
============================
```

## Help

```
$ sourcepredict -h
usage: SourcePredict v0.2 [-h] [-a ALPHA] [-s SOURCES] [-l LABELS]
                          [-n NORMALIZATION] [-o OUTPUT] [-u UMAP] [-se SEED]
                          [-k KFOLD] [-t THREADS]
                          otu_table

==========================================================
SourcePredict v0.2
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
                    data/modern_gut_microbiomes_sources.csv
  -l LABELS         Path to labels csv file. Default =
                    data/modern_gut_microbiomes_labels.csv
  -n NORMALIZATION  Normalization method (RLE | CLR | Subsample). Default =
                    RLE
  -o OUTPUT         Output file basename. Default =
                    <sample_basename>.sourcepredict.csv
  -u UMAP           Umap embedding csv file. Default = None
  -se SEED          Seed for random generator. Default = 42
  -k KFOLD          Number of fold for K-fold cross validation in feature
                    selection and parameter optimization. Default = 3
  -t THREADS        Number of threads for parallel processing. Default = 2
```

## Sourcepredict source file

- The sources were obtained with the [Kraken based pipeline](utils/kraken_pipeline/kraken_pipe.nf) included in this repository, using the [*MiniKraken2_v2_8GB*](https://ccb.jhu.edu/software/kraken2/dl/minikraken2_v2_8GB.tgz).  
- The default source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_sources.csv)
- The label file for this source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_labels.csv)


### Current species included in the source file

- *Sus scrofa*
- *Homo sapiens*
- *Canis familiaris*

### Updating the source file 

To update the sourcefile with new kraken results, see the instruction in the [dedicated Jupyter notebook](notebooks/merge_new_data.ipynb) 
