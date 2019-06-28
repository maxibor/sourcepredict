[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict) [![Coverage Status](https://coveralls.io/repos/github/maxibor/sourcepredict/badge.svg?branch=master)](https://coveralls.io/github/maxibor/sourcepredict?branch=master) [![Anaconda-Server Badge](https://anaconda.org/maxibor/sourcepredict/badges/installer/conda.svg)](https://conda.anaconda.org/maxibor) [![Documentation Status](https://readthedocs.org/projects/sourcepredict/badge/?version=latest)](https://sourcepredict.readthedocs.io/en/latest/?badge=latest)


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

## Documentation

The documentation of SourcePredict is available here: [sourcepredict.readthedocs.io](https://sourcepredict.readthedocs.io/en/latest/)

## Sourcepredict source file

- The sources were obtained with the [Kraken based pipeline](utils/kraken_pipeline/kraken_pipe.nf) included in this repository, using the [*MiniKraken2_v2_8GB*](https://ccb.jhu.edu/software/kraken2/dl/minikraken2_v2_8GB.tgz).  
- The default source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_sources.csv)
- The label file for this source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_labels.csv)


### Environments included in the default source file

- *Homo sapiens* gut microbiome
- *Canis familiaris* gut microbiome
- Soil microbiome

### Updating the source file 

To update the sourcefile with new kraken results, see the instruction in the [dedicated Jupyter notebook](notebooks/merge_new_data.ipynb) 
