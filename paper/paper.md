---
title: 'Sourcepredict: Prediction of metagenomic sample sources using dimension reduction followed by machine learning classification'
tags:
  - microbiome
  - source tracking
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

SourcePredict is a Python package distributed through Conda, to classify and predict the origin of metagenomic samples, given a reference dataset of known origins, a problem also known as source tracking.

DNA shotgun sequencing of human, animal, and environmental samples has opened up new doors to explore the diversity of life in these different environments, a field known as metagenomics [@metagenomics]. One aspect of metagenomics is investigating the community composition of organisms within a sequencing sample with tools known as taxonomic classifiers, such as Kraken [@kraken].

In cases where the origin of a metagenomic sample, its source, is unknown, it is often part of the research question to predict and/or confirm the source. For example, in microbial archaelogy, it is sometimes necessary to rely on metagenomics to validate the source of paleofaeces.
Using samples of known sources, a reference dataset can be established with the taxonomic composition of the samples, i.e., the organisms identified in the samples as features, and the sources of the samples as class labels.

With this reference dataset, a machine learning algorithm can be trained to predict the source of unknown samples (sinks) from their taxonomic composition.

Other tools used to perform the prediction of a sample source already exist, such as SourceTracker [@sourcetracker], which employs Gibbs sampling.

However, the Sourcepredict results are more easily interpreted since the samples are embedded in a human observable low-dimensional space. This embedding is performed by a dimension reduction algorithm followed by K-Nearest-Neighbours (KNN) classification.

## Method

Starting with a numerical organism count matrix (samples as columns, organisms as rows, obtained by a taxonomic classifier) of merged references and sinks datasets, samples are first normalized relative to each other, to correct for uneven sequencing depth using the geometric mean of pairwise ratios (GMPR) method (default) [@gmpr].

After normalization, Sourcepredict performs a two-step prediction algorithm. First, it predicts the proportion of unknown sources, i.e., which are not represented in the reference dataset. Second, it predicts the proportion of each known source of the reference dataset in the sink samples.

Organisms are represented by their taxonomic identifiers (TAXID).

### Prediction of the proportion of unknown sources

Let $S_i \in \{S_1, .., S_n\}$ be a sample from the normalized sinks dataset $D_{sink}$, $o_{j}^{\ i} \in \{o_{1}^{\ i},.., o_{n_o^{\ i}}^{\ i}\}$  an organism in $S_i$, and $n_o^{\ i}$  the total number of organisms in $S_i$, with $o_{j}^{\ i} \in \mathbb{Z}+$. Let $m$ be the mean number of samples per source in the reference dataset, such that $m = \frac{1}{O}\sum_{i=1}^{O}S_i$. For each $S_i$ sample, I define $||m||$ derivative samples $U_k^{S_i} \in \{U_1^{S_i}, ..,U_{||m||}^{S_i}\}$ to add to the reference dataset to account for the unknown source proportion in a test sample. Separately for each $S_i$, a proportion denoted $\alpha \in [0,1]$ (default = $0.1$) of each $o_{j}^{\ i}$ organism of $S_i$ is added to each $U_k^{S_i}$ sample such that $U_k^{S_i}(o_j^{\ i}) = \alpha \cdot x_{i \ j}$ , where $x_{i \ j}$ is sampled from a Gaussian distribution $\mathcal{N}\big(S_i(o_j^{\ i}), 0.01)$. The $||m||$ $U_k^{S_i}$ samples are then added to the reference dataset $D_{ref}$, and labeled as *unknown*, to create a new reference dataset denoted ${}^{unk}D_{ref}$. To predict the proportion of unknown sources, a Bray-Curtis [@bray-curtis] pairwise dissimilarity matrix of all $S_i$ and $U_k^{S_i}$ samples is computed using scikit-bio [@scikit-bio]. This distance matrix is then embedded in two dimensions (default) with the scikit-bio implementation of PCoA. This sample embedding is divided into three subsets: ${}^{unk}D_{train}$ ($64\%$), ${}^{unk}D_{test}$ ($20\%$), and ${}^{unk}D_{validation}$($16\%$). The scikit-learn [@scikit-learn] implementation of KNN algorithm is then trained on ${}^{unk}D_{train}$, and the training accuracy is computed with ${}^{unk}D_{test}$. This trained KNN model is then corrected for probability estimation of the unknown proportion using the scikit-learn implementation of Platt's scaling method [@platt] with ${}^{unk}D_{validation}$. The proportion of unknown sources in $S_i$, $p_u \in [0,1]$ is then estimated using this trained and corrected KNN model. Ultimately, this process is repeated independently for each sink sample $S_i$ of $D_{sink}$.

### Prediction of the proportion of known sources

First, only organism TAXIDs corresponding to the species taxonomic level are retained using the ETE toolkit [@ete3]. A weighted Unifrac (default) [@wu] pairwise distance matrix is then computed on the merged and normalized training dataset $D_{ref}$ and test dataset $D_{sink}$ with scikit-bio, using the NCBI taxonomy as a reference tree. This distance matrix is then embedded in two dimensions (default) using the scikit-learn implementation of t-SNE [@tsne]. The 2-dimensional embedding is then split back to training ${}^{tsne}D_{ref}$ and testing dataset ${}^{tsne}D_{sink}$. The KNN algorithm is then trained on the train subset, with a five (default) cross validation to look for the optimum number of K-neighbors.
The training dataset ${}^{tsne}D_{ref}$ is further divided into three subsets: ${}^{tsne}D_{train}$ ($64\%$), ${}^{tsne}D_{test}$ ($20\%$), and ${}^{tsne}D_{validation}$ ($16\%$). The training accuracy is then computed with ${}^{tsne}D_{test}$. Finally, this second trained KNN model is also corrected for source proportion estimation using the scikit-learn implementation of the Platt's method with ${}^{tsne}D_{validation}$. The proportion $p_{c_s} \in [0,1]$ of each of the $n_s$ sources $c_s \in \{c_{1},\ ..,\ c_{n_s}\}$ in each sample $S_i$ is then estimated using this second trained and corrected KNN model.

### Combining unknown and source proportions

For each sample $S_i$ of the test dataset $D_{sink}$, the predicted unknown proportion $p_{u}$ is then combined with the predicted proportion $p_{c_s}$ for each of the $n_s$ sources $c_s$ of the training dataset such that $\sum_{c_s=1}^{n_s} s_c + p_u = 1$ where $s_c = p_{c_s} \cdot p_u$.

Finally, a summary table gathering the estimated sources proportions is returned as a `csv` file, as well as the t-SNE embedding sample coordinates.

## Acknowledgements

Thanks to Dr.\ Christina Warinner, Dr.\ Alexander Herbig, Dr.\ AB Rohrlach, and Alexander Hübner for their valuable comments and for proofreading this manuscript.
This work was funded by the Max Planck Society and the Deutsche Forschungsgemeinschaft, project code: EXC 2051 #390713860.

# References
