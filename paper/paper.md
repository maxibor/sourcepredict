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


Let $S_i \in \{S_1, .., S_n\}$ be a sample of size $O$ organisms $o_j$ from the test dataset $D_{sink}$, with $o_j \in \mathbb{Z}+$, and $j\in[1,O]$.  
Let $m$ be the mean number of samples per class in the reference dataset, such as $m = \frac{1}{O}\sum_{i=1}^{O}S_i$.  
I define $|m|$ estimated samples $U_k$ to add to the training dataset to account for the unknown source proportion in a test sample, with $k \in \{1,..,|m|\}$.  

To compute each $U_k$, a $\alpha$ proportion ($\alpha \in [0,1]$, default = $0.1$) of each $o_j$ organism is added to the training dataset for each $U_k$ samples, such that $U_k(o_j) = \alpha \cdot x_{i \ j}$ , where $x_{i \ j}$ is sampled from the Gaussian distribution $\mathcal{N}\big(\mu=S_i(o_j), \sigma=0.1\big)$.  

The $|m|$ $U_k$ samples are then merged as columns to the reference dataset $D_{ref}$ (samples in columns, organisms as rows) to create a new reference dataset denoted $D_{ref\ u}$

To predict this unknown proportion, the dimension of the reference dataset $D_{ref\ u}$ is reduced to the first 20 principal components with the scikit-learn [@scikit-learn] implementation of  PCA.  
This dimensionally reduced reference dataset is further divided into three subsets: $D_{train\ u}$ ($64\%$), $D_{test\ u}$ ($20\%$), and $D_{validation\ u}$($16\%$). 
 
The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm is then trained on $D_{train\ u}$, and the test accuracy is computed with $D_{test\ u}$ .  
This trained KNN model is then corrected for probability estimation of unknown proportion using the scikit-learn implementation of the Platt's scaling method [@platt] with $D_{validation\ u}$.
This procedure is repeated for each $S_i$ sample of the test dataset  $D_{sink}$.

$p_u$ is then estimated using this trained and corrected KNN mode, where $p_u$ is the proportion of unknown sources in each $S_i$ sample. 

### Prediction of known source proportion

First, only organism TAXID corresponding to the *species* taxonomic level are kept using ETE toolkit [@ete3].
A distance matrix is then computed on the merged training dataset $D_{ref}$ and test dataset $D_{sink}$ using the scikit-bio implementation of weighted Unifrac distance (default) [@wu].

The distance matrix is embedded in two dimensions using the scikit-learn implementation of t-SNE [@tsne].

The 2-dimensional embedding is then split back to training $D_{ref\ t}$ and testing dataset $D_{sink\ t}$.

The training dataset $D_{ref\ tsne}$ is further divided into three subsets: $D_{train\ t}$ ($64\%$), $D_{test\ t}$ ($20\%$), and $D_{validation\ t}$ ($16\%$).  
The KNN algorithm is then trained on the train subset, and the test accuracy is computed with $D_{test\ t}$.  
This trained KNN model is then corrected for source proportion estimation using the scikit-learn implementation of the Platt's method with $D_{validation\ t}$.

$p_{c}$ is then estimated using this trained and corrected KNN model, where $p_{c}$ is the proportion of each of source $c$ in each sample $S_i$.

### Combining unknown and source proportion

Finally, for each sample $S_i$ of the test dataset $D_{sink}$, the predicted unknown proportion $p_{u}$ is then combined with the predicted proportion $p_{c}$ for each of the $C$ sources $c$ of the training dataset such that $\sum_{c=1}^{C} s_c + p_u = 1$ where $s_c = p_c \cdot p_u$.

Finally, a summary table is created to gather the estimated sources proportions.

## Acknowledgements

Thanks to Dr. Alexander Herbig and Dr. Adam Ben Rohrlach for their valuable comments and for proofreading this manuscript.

# References
