[![Build Status](https://travis-ci.com/maxibor/sourcepredict.svg?token=pwT9AgYi4qJY4LTp9WUy&branch=master)](https://travis-ci.com/maxibor/sourcepredict) [![Coverage Status](https://coveralls.io/repos/github/maxibor/sourcepredict/badge.svg?branch=master)](https://coveralls.io/github/maxibor/sourcepredict?branch=master) [![Anaconda-Server Badge](https://anaconda.org/maxibor/sourcepredict/badges/installer/conda.svg)](https://conda.anaconda.org/maxibor) [![Documentation Status](https://readthedocs.org/projects/sourcepredict/badge/?version=latest)](https://sourcepredict.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10.5281/zenodo.3379603.svg)](https://doi.org/10.5281/zenodo.10.5281/zenodo.3379603)
 [![DOI](https://joss.theoj.org/papers/10.21105/joss.01540/status.svg)](https://doi.org/10.21105/joss.01540)

---

<img src="https://raw.githubusercontent.com/maxibor/sourcepredict/master/img/sourcepredict_logo.png" width="300">

Sourcepredict is a Python package distributed through Conda, to classify and predict the origin of metagenomic samples, given a reference dataset of known origins, a problem also known as source tracking.
Sourcepredict solves this problem by using machine learning classification on dimensionally reduced datasets.

## Installation

With conda (recommended)

```bash
$ conda install -c conda-forge -c maxibor sourcepredict
```

With pip

```bash
$ pip install sourcepredict
```

## Example

### Input

- Sink taxonomic count file (see [example file](https://github.com/maxibor/sourcepredict/blob/master/data/test/dog_test_sink_sample.csv) and [documentation](https://sourcepredict.readthedocs.io/en/latest/usage.html#sink_table))
- Source taxonomic count file (see [example file](https://github.com/maxibor/sourcepredict/blob/master/data/modern_gut_microbiomes_sources.csv) and [documentation](https://sourcepredict.readthedocs.io/en/latest/usage.html#s-sources))
- Source label file (see [example file](https://github.com/maxibor/sourcepredict/blob/master/data/modern_gut_microbiomes_labels.csv) and [documentation](https://sourcepredict.readthedocs.io/en/latest/usage.html#l-labels))

### Usage 

```bash
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/test/dog_test_sink_sample.csv -O dog_example.csv
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/modern_gut_microbiomes_labels.csv -O sp_labels.csv
$ wget https://raw.githubusercontent.com/maxibor/sourcepredict/master/data/modern_gut_microbiomes_sources.csv -O sp_sources.csv
$ sourcepredict -s sp_sources.csv -l sp_labels.csv dog_example.csv
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

## Sourcepredict example files

- The sources were obtained with a simple [Nextflow pipeline](https://github.com/maxibor/kraken-nf), with Kraken2 using the [*MiniKraken2_v2_8GB*](https://ccb.jhu.edu/software/kraken2/dl/minikraken2_v2_8GB.tgz).  
See the [documentation](https://sourcepredict.readthedocs.io/en/latest/custom_sources.html) for more informations on how to build a custom source file. 
- The example source file is here [modern_gut_microbiomes_sources.csv](https://github.com/maxibor/sourcepredict/raw/master/data/modern_gut_microbiomes_sources.csv)
- The example label file is here [modern_gut_microbiomes_sources.csv](https://github.com/maxibor/sourcepredict/raw/master/data/modern_gut_microbiomes_labels.csv)


### Environments included in the example source file

- *Homo sapiens* gut microbiome ([1](https://doi.org/10.1038/nature11234), [2](https://doi.org/10.1093/gigascience/giz004), [3](https://doi.org/10.1038/s41564-019-0409-6), [4](https://doi.org/10.1016/j.cell.2019.01.001), [5](https://doi.org/10.1038/ncomms7505), [6](http://doi.org/10.1016/j.cub.2015.04.055))
- *Canis familiaris* gut microbiome ([1](https://doi.org/10.1186/s40168-018-0450-3))
- Soil microbiome ([1](https://doi.org/10.1073/pnas.1215210110), [2](https://www.ncbi.nlm.nih.gov/bioproject/?term=322597), [3](https://dx.doi.org/10.1128%2FAEM.01646-17))

## Contributing Code, Documentation, or Feedback

If you wish to contribute to Sourcepredict, you are welcome and encouraged to contribute by opening an issue, or creating a pull-request. All contributions will be made under the GPLv3 license. More informations can found on the [contributing page](https://github.com/maxibor/sourcepredict/blob/master/contributing.md).

## How to cite

Sourcepredict has been published in [JOSS](https://joss.theoj.org/papers/10.21105/joss.01540).

```
@article{Borry2019Sourcepredict,
	journal = {Journal of Open Source Software},
	doi = {10.21105/joss.01540},
	issn = {2475-9066},
	number = {41},
	publisher = {The Open Journal},
	title = {Sourcepredict: Prediction of metagenomic sample sources using dimension reduction followed by machine learning classification},
	url = {http://dx.doi.org/10.21105/joss.01540},
	volume = {4},
	author = {Borry, Maxime},
	pages = {1540},
	date = {2019-09-04},
	year = {2019},
	month = {9},
	day = {4}
}
```