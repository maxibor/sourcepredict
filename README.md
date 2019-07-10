[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict) [![Coverage Status](https://coveralls.io/repos/github/maxibor/sourcepredict/badge.svg?branch=master)](https://coveralls.io/github/maxibor/sourcepredict?branch=master) [![Anaconda-Server Badge](https://anaconda.org/maxibor/sourcepredict/badges/installer/conda.svg)](https://conda.anaconda.org/maxibor) [![Documentation Status](https://readthedocs.org/projects/sourcepredict/badge/?version=latest)](https://sourcepredict.readthedocs.io/en/latest/?badge=latest)


<img src="img/sourcepredict_logo.png" width="300">

Sourcepredict is a Python package distributed through Conda, to classify and predict the origin of metagenomic samples, given a reference dataset of known origins, a problem also known as source tracking.
Sourcepredict solves this problem by using machine learning classification on dimensionally reduced datasets.

## Installation

```
$ conda install -c conda-forge -c etetoolkit -c bioconda -c maxibor sourcepredict
```

## Example

### Input

- Sink file OTU count file (see [example file](https://github.com/maxibor/sourcepredict/blob/master/data/test/dog_test_sample.csv) and [documentation](https://sourcepredict.readthedocs.io/en/latest/usage.html#otu_table))
- Source file OTU count file (see [example file](https://github.com/maxibor/sourcepredict/blob/master/data/modern_gut_microbiomes_sources.csv) and [documentation](https://sourcepredict.readthedocs.io/en/latest/usage.html#s-sources))
- Source label file (see [example file](https://github.com/maxibor/sourcepredict/blob/master/data/modern_gut_microbiomes_labels.csv) and [documentation](https://sourcepredict.readthedocs.io/en/latest/usage.html#l-labels))

### Usage 

```bash
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/test/dog_test_sample.csv -O dog_test_sample.csv
$ sourcepredict dog_test_sample.csv
Step 1: Checking for unknown proportion
  == Sample: ERR1915662 ==
	Adding unknown
	Normalizing (GMPR)
	Computing Bray-Curtis distance
	Performing MDS embedding in 2 dimensions
	KNN machine learning
	Training KNN classifier on 2 cores...
	-> Testing Accuracy: 1.0
	----------------------
	- Sample: ERR1915662
		 known:98.61%
		 unknown:1.39%
Step 2: Checking for source proportion
	Computing weighted_unifrac distance on species rank
	TSNE embedding in 2 dimensions
	KNN machine learning
	Performing 5 fold cross validation on 2 cores...
	Trained KNN classifier with 10 neighbors
	-> Testing Accuracy: 0.99
	----------------------
	- Sample: ERR1915662
		 Canis_familiaris:96.1%
		 Homo_sapiens:2.47%
		 Soil:1.43%
Sourcepredict result written to dog_test_sample.sourcepredict.csv
```

### Output

Sourcepredict output the predicted source contribution to each sink sample, and the embedding of all samples in the lower dimensional space.  See [documentation](https://sourcepredict.readthedocs.io/en/latest/results.html) for details.

### Runtime

Depending on the normalization method (`-n`), the embedding (`-me`) method, the cpus available for parallel processing (`-t`), and the data, the runtime should be between a few seconds and a few minutes per sink sample.


## Documentation

The documentation of SourcePredict is available here: [sourcepredict.readthedocs.io](https://sourcepredict.readthedocs.io/en/latest/)

## Sourcepredict source file

- The sources were obtained with the [Kraken based pipeline](utils/kraken_pipeline/kraken_pipe.nf) included in this repository, using the [*MiniKraken2_v2_8GB*](https://ccb.jhu.edu/software/kraken2/dl/minikraken2_v2_8GB.tgz).  
- The default source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_sources.csv)
- The label file for this source file is here [data/modern_gut_microbiomes_sources.csv](data/modern_gut_microbiomes_labels.csv)


### Environments included in the default source file

- *Homo sapiens* gut microbiome ([1](https://doi.org/10.1038/nature11234), [2](https://doi.org/10.1093/gigascience/giz004), [3](https://doi.org/10.1038/s41564-019-0409-6), [4](https://doi.org/10.1016/j.cell.2019.01.001), [5](https://doi.org/10.1038/ncomms7505), [6](http://doi.org/10.1016/j.cub.2015.04.055))
- *Canis familiaris* gut microbiome ([1](https://doi.org/10.1186/s40168-018-0450-3))
- Soil microbiome ([1](https://doi.org/10.1073/pnas.1215210110), [2](https://www.ncbi.nlm.nih.gov/bioproject/?term=322597), [3](https://dx.doi.org/10.1128%2FAEM.01646-17))

### Updating the source file 

To update the sourcefile with new kraken results, see the instruction in the [dedicated Jupyter notebook](notebooks/merge_new_data.ipynb) 
