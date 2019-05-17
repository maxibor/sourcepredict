---
title: 'Sourcepredict: Prediction/source tracking of metagenomic sample sources using machine learning'
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
date: 15th May 2019
bibliography: paper.bib
---

# Summary

SourcePredict [(github.com/maxibor/sourcepredict)](https://github.com/maxibor/sourcepredict) is a Python Conda package to classify and predict the source of metagenomics sample given a reference dataset of known sources.  

DNA shotgun sequencing of human, animal, and environmental samples opened up new doors to explore the diversity of life in these different environments, a field known as metagenomics [@metagenomics].  
One of the aspect of metagenomics is to look at the organism composition in a sequencing sample, with tools known as taxonomic classifiers.
These taxonomic classifiers, such as Kraken [@kraken] for example, will compute the organism taxonomic composition, from the DNA sequencing data.

When in most cases the origin (source) of a metagenomic sample is known, it is sometimes part of the research question to infer and/or confirm its source.
Using samples of known sources, a reference dataset can be established with the samples taxonomic composition (the organisms identified in the sample) as features, and the source of the sample as class labels.
With this reference dataset, a machine learning algorithm can be trained to predict the source of unlabeled samples from their taxonomic composition.  
Compared to SourceTracker [@sourcetracker], which uses gibbs sampling, Sourcepredict uses dimension reduction algorithms, followed by K-Nearest-Neighbors (KNN) classification.

Here, I present SourcePredict for the classification/prediction of unlabeled sample sources from their taxonomic compositions.

## Method

All samples are first normalized to correct for uneven sequencing depth using GMPR (default) [@gmpr].
After normalization, Sourcepredict performs a two steps prediction: first a prediction of the proportion of unknown sources, i.e. not represented in the reference dataset. Then a prediction of the proportion of each known source of the reference dataset in the test samples.

Organism are represented by their taxonomic identifiers (TAXID).

### Prediction of unknown sources proportion

Let $S$ be a sample of size $O$ organims from the test dataset $D_{sink}$  
Let $n$ be the average number of samples per class in the reference dataset.  
I define $U_n$ samples to add to the training dataset to account for the unknown source proportion in a test sample.  

To compute $U_n$, a $\alpha$ proportion (default = $0.1$) of each $o_i$ organism (with $i\in[1,O]$) is added to the training dataset for each $U_j$ samples (with $j\in[1,n]$), such as $U_j(o_i) = \alpha\times S(o_i)$  

The $U_n$ samples are then merged as columns to the reference dataset ($D_{ref}$) to create a new reference dataset denoted $D_{ref\ unknown}$

To predict this unknown proportion, the dimension of the reference dataset $D_{ref\ unknown}$ (samples in columns, organisms as rows) is first reduced to 20 with the scikit-learn [@scikit-learn] implementation of  PCA.  
This reference dataset is then divided into three subsets: $D_{train\ unknown}$ (64%), $D_{test\ unknown}$ (20%), and $D_{validation\ unknown}$(16%). 
 
The scikit-learn implementation of KNN algorithm is then trained on $D_{train\ unknown}$, and the test accuracy is computed with $D_{test\ unknown}$ .  
The trained KNN model is then corrected for probability estimation of unknown proportion using the scikit-learn implementation of the Platt's scaling method [@platt] with $D_{validation\ unknown}$.
This procedure is repeated for each sample of the test dataset.

The proportion of unknown $p_{unknown}$ sources in each sample is then computed using the trained and corrected KNN model.

### Prediction of known source proportion

First, only organism TAXID corresponding to the *species* taxonomic level are kept using ETE toolkit [@ete3].
A distance matrix is then computed on the merged training dataset $D_{ref}$ and test dataset $D_{sink}$ using the scikit-bio implementation of weighted Unifrac distance (default) [@wu].

The distance matrix is then embedded in two dimensions using the scikit-learn implementation of t-SNE [@tsne].

The 2-dimensional embedding is then split back to training $D_{ref\ tsne}$ and testing dataset $D_{sink\ tsne}$.

The training dataset $D_{ref\ tsne}$ is further divided into three subsets: $D_{train\ tsne}$ (64%), $D_{test\ tsne}$ (20%), and $D_{validation\ tsne}$ (16%).  
The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm is then trained on the train subset, and the test accuracy is computed with $D_{test\ tsne}$.  
The trained KNN model is then corrected for source proportion estimation using the scikit-learn implementation of the Platt's method with $D_{validation\ tsne}$.

The proportion of each source $p_{c}$ sources in each sample is then computed using the trained and corrected KNN model.

### Combining unknown and source proportion

Finally, for each sample, the predicted unknown proportion $p_{unknown}$ is then combined with the predicted proportion $p_{c}$ of each of the $C$ source class $c$ of the training dataset such as:

$$\sum_{c=1}^{C} s_c + p_{unknown} = 1$$

with  

$$s_c = p_{c}\times p_{unknown}$$

Finally, a summary table is created to gather the estimated sources proportions.

## Acknowledgements

Thanks to Dr. Alexander Herbig for proofreading this manuscript.

# References
