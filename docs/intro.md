# Introduction

![](_assets/img/sourcepredict_logo.png)


Prediction/source tracking of metagenomic samples source using machine learning 

----

SourcePredict [(github.com/maxibor/sourcepredict)](https://github.com/maxibor/sourcepredict) is a Python package distributed through Conda, to classify and predict the origin of metagenomic samples, given a reference dataset of known origins, a problem also known as source tracking.  
DNA shotgun sequencing of human, animal, and environmental samples has opened up new doors to explore the diversity of life in these different environments, a field known as metagenomics.  
One aspect of metagenomics is investigating the community composition of organisms within a sequencing sample with tools known as taxonomic classifiers, such as [Kraken](https://ccb.jhu.edu/software/kraken/).

In cases where the origin of a metagenomic sample, its source, is unknown, it is often part of the research question to predict and/or confirm the source.
For example, in microbial archaelogy, it is sometimes necessary to rely on metagenomics to validate the source of paleofaeces.
Using samples of known sources, a reference dataset can be established with the taxonomic composition of the samples, *i.e.* the organisms identified in the samples as features, and the sources of the samples as class labels.
With this reference dataset, a machine learning algorithm can be trained to predict the source of unknown samples (sinks) from their taxonomic composition.  
Other tools used to perform the prediction of a sample source already exist, such as SourceTracker [sourcetracker](https://www.nature.com/articles/nmeth.1650), which employs Gibbs sampling.  
However, the Sourcepredict results are easier interpreted since the samples are embedded in a human observable low-dimensional space. This embedding is performed by a dimension reduction algorithm followed by K-Nearest-Neighbours (KNN) classification.