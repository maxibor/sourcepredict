[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict) [![Anaconda-Server Badge](https://anaconda.org/maxibor/sourcepredict/badges/installer/conda.svg)](https://conda.anaconda.org/maxibor)

<img src="img/sourcepredict_logo.png" width="300">

Prediction/classification of the origin of a metagenomics sample.

## Installation

```
$ conda install -c etetoolkit -c bioconda -c maxibor sourcepredict
```

## Example

```bash
$ wget wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/test/dog_test_sample.csv -O dog_test_sample.csv
$ sourcepredict -t 6 dog_test_sample.csv
Step 1: Checking for unknown proportion
  == Sample: ERR1915662 ==
	Adding unknown
	Normalizing (GMPR)
	Computing Bray-Curtis distance
	Performing MDS embedding in 2 dimensions
	KNN machine learning
	Training KNN classifier on 6 cores...
	-> Testing Accuracy: 1.0
	----------------------
	- Sample: ERR1915662
		 known:98.61%
		 unknown:1.39%
  == Sample: ERR1915662_copy ==
	Adding unknown
	Normalizing (GMPR)
	Computing Bray-Curtis distance
	Performing MDS embedding in 2 dimensions
	KNN machine learning
	Training KNN classifier on 6 cores...
	-> Testing Accuracy: 1.0
	----------------------
	- Sample: ERR1915662_copy
		 known:98.61%
		 unknown:1.39%
Step 2: Checking for source proportion
	Computing weighted_unifrac distance on species rank
	TSNE embedding in 2 dimensions
	KNN machine learning
	Performing 5 fold cross validation on 6 cores...
	Trained KNN classifier with 10 neighbors
	-> Testing Accuracy: 0.99
	----------------------
	- Sample: ERR1915662
		 Canis_familiaris:96.14%
		 Homo_sapiens:2.44%
		 Soil:1.42%
	- Sample: ERR1915662_copy
		 Canis_familiaris:96.14%
		 Homo_sapiens:2.44%
		 Soil:1.42%
Sourcepredict result written to dog_test_sample.sourcepredict.csv
```

## Help

```
$ sourcepredict -h
usage: SourcePredict v0.3.1 [-h] [-a ALPHA] [-s SOURCES] [-l LABELS]
                            [-n NORMALIZATION] [-dt DISTANCE] [-me METHOD]
                            [-e EMBED] [-di DIM] [-o OUTPUT] [-se SEED]
                            [-k KFOLD] [-t THREADS]
                            otu_table

==========================================================
SourcePredict v0.3.1
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
  -n NORMALIZATION  Normalization method (RLE | CLR | Subsample | GMPR).
                    Default = GMPR
  -dt DISTANCE      Distance method. (unweighted_unifrac | weighted_unifrac)
                    Default = weighted_unifrac
  -me METHOD        Embedding Method. TSNE or UMAP. Default = TSNE
  -e EMBED          Output embedding csv file. Default = None
  -di DIM           Number of dimensions to retain for dimension reduction.
                    Default = 2
  -o OUTPUT         Output file basename. Default =
                    <sample_basename>.sourcepredict.csv
  -se SEED          Seed for random generator. Default = 42
  -k KFOLD          Number of fold for K-fold cross validation in feature
                    selection and parameter optimization. Default = 5
  -t THREADS        Number of threads for parallel processing. Default = 2
```

## Sourcepredict source file

- The sources were obtained with the [Kraken based pipeline](utils/kraken_pipeline/kraken_pipe.nf) included in this repository, using the [*MiniKraken2_v2_8GB*](https://ccb.jhu.edu/software/kraken2/dl/minikraken2_v2_8GB.tgz).  
- The default source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_sources.csv)
- The label file for this source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_labels.csv)


### Environments included in the default source file

- *Homo sapiens* gut microbiome
- *Canis familiaris* gut microbiom
- Soil microbiome

### Updating the source file 

To update the sourcefile with new kraken results, see the instruction in the [dedicated Jupyter notebook](notebooks/merge_new_data.ipynb) 
