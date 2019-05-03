---
title: 'Sourcepredict: Prediction/source tracking of metagenomic samples source using machine learning'
tags:
  - microbiome
  - sourcetracking
  - machine learning
authors:
 - name: Maxime Borry
   orcid: 0000-0001-9140-7559
   affiliation: "1"
affiliations:
 - name: Department of Archaeogenetics, Max Planck Institute for the Science of Human History, Jena, 07745, Germany
   index: 1
date: 3rd May 2019
bibliography: paper.bib
---

# Summary

SourcePredict [(github.com/maxibor/sourcepredict)](https://github.com/maxibor/sourcepredict) is a Python package to classify and predict the source of metagenomics sample given a training set.  

The DNA shotgun sequencing of human, animal, and environmental samples opened up new doors to explore the diversity of life in these different environments, a field known as metagenomics [@metagenomics].  
One of the goals of metagenomics is to look at the composition of a sequencing sample with tools known as taxonomic classifiers.
These taxonomic classifiers, such as Kraken [@kraken] for example, will compute the taxonomic composition in Operational Taxonomic Unit (OTU), from the DNA sequencing data.

When in most cases the origin of a metagenomic sample is known, it is sometimes part of the research question to infer and/or confirm its source.  
Using samples of known sources, a training set can be established with the OTU sample composition as features, and the source of the sample as class labels.  
With this training set, a machine learning algorithm can be trained to predict the source of unlabeled samples from their OTU taxonomic composition.

Here, I developed SourcePredict to perform the classification/prediction of unlabeled samples sources from their OTU taxonomic compositions.

## Method

All samples are first normalized to correct for uneven sequencing depth using GMPR (default) [@gmpr].
After normalization, Sourcepredict performs a two steps prediction.

### Prediction of unknown sources proportion

The unknown sources proportion is the proportion of OTUs in the test sample which are not present in the training dataset.  

Let $S$ be a sample of size $O$ with $O$ OTUs from the test dataset $D_{test}$  
Let $n$ be the average number of samples per class in the training dataset.  
Let $U_n$ be the samples to add to the training dataset to account for the unknown source proportion in a test sample.  

First a $\alpha$ proportion (default=$0.1$) of each $o_i$ OTU (with $i\in[1,O]$) is added to the training dataset for each $U_j$ samples (with $j\in[1,n]$), such as $U_j(o_i) = \alpha\times S_(o_i)$  

The $U_n$ samples are then merged as columns to the training dataset ($D_{train}$) to create a new training dataset denoted $D_{train\ unknown}$

To predict this unknown proportion, the dimension of the training dataset $D_{train\ unknown}$ (samples in columns, OTUs as rows) is first reduced to 20 with the scikit-learn [@scikit-learn] implementation of the PCA.  
This training dataset is further divided into three subsets: train (64%), test (20%), and validation (16%).  
The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm is then trained on the train subset, and the test accuracy is computed with the test subset.  
The trained KNN model is then corrected for probability estimation of unknown proportion using the scikit-learn implementation of the Platt's scaling method [@platt] with the validation subset.
This procedure is repeated for each sample of the test dataset.

### Prediction of known source proportion

First, only OTUs corresponding to the *species* taxonomic level are kept using ETE toolkit [@ete3].
A distance matrix is then computed on the merged training dataset $D_{train}$ and test dataset $D_{test}$ using the scikit-bio implementation of weighted Unifrac distance (default) [@wu].

The distance matrix is then embedded in two dimensions using the scikit-learn implementation of t-SNE [@tsne].

The 2-dimensional embedding is then split back to training and testing dataset.

The training dataset is further divided into three subsets: train (64%), test (20%), and validation (16%).  
The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm is then trained on the train subset, and the test accuracy is computed with the test subset.  
The trained KNN model is then corrected for source proportion estimation using the scikit-learn implementation of the Platt's method with the validation subset.

### Combining unknown and source proportion

For each sample, the predicted unknown proportion $p\_{unknown}$ is then combined with the predicted proportion of each of the $C$ source class $c$ of the training dataset such as:

$$\sum_{c=1}^{C} s_c + p_{unknown} = 1$$

with  

$$s_c = s_{c\ predicted}\times p_{unknown}$$

## CLI

The SourcePredict CLI is handled with ArgParse. A typical command to use SourcePredict is as simple as:  

`sourcepredict path/to/test_otu_table.csv`

The documentation of CLI is available at [sourcepredict.readthedocs.io](https://sourcepredict.readthedocs.io)

# References
