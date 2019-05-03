# Introduction

![](_assets/img/sourcepredict_logo.png)


Prediction/source tracking of metagenomic samples source using machine learning 

----

[SourcePredict](https://github.com/maxibor/sourcepredict) is a Python package to classify and predict the source of metagenomics sample given a training set.  

The DNA shotgun sequencing of human, animal, and environmental samples opened up new doors to explore the diversity of life in these different environments, a field known as metagenomics.  
One of the goals of metagenomics is to look at the composition of a sequencing sample with tools known as taxonomic classifiers.
These taxonomic classifiers, such as Kraken for example, will compute the taxonomic composition in Operational Taxonomic Unit (OTU), from the DNA sequencing data.

When in most cases the origin of a metagenomic sample is known, it is sometimes part of the research question to infer and/or confirm its source.  
Using samples of known sources, a training set can be established with the OTU sample composition as features, and the source of the sample as class labels.  
With this training set, a machine learning algorithm can be trained to predict the source of unlabeled samples from their OTU taxonomic composition.

SourcePredict performs the classification/prediction of unlabeled samples sources from their OTU taxonomic compositions.
