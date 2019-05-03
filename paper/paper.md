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
 - name: Department of Archaegenetics, Max Planck Institute for the Science of Human Histoy, Jena, 07745, Germany
   index: 1
date: 2 May 2019
bibliography: paper.bib
---

# Summary

SourcePredict is a Python package to classify and predict the source of metagenomics sample given a training set.
The DNA shotgun sequencing of human, animal, and environmental samples opened up new doors to explore the diversity of life in these different environments, a field known as metagenomics [ADD_REF].
One of the goal of metagenomics is to look at the composition of a sequencing sample with tools known as taxonomic classifiers.
These taxonomic classifiers, such as Kraken [@kraken] for example, will compute the organism composition from sequencing data.

When in most cases the origin of a metagenomics sample is known, it is sometimes part of the research question to infer and/or confirm its origin.
For samples of known origin, a training set can be established with the sample composition as data, and the origin of the sample as labels.
Using this training set, a machine learning algorithm can the predict the origin of unlabeled samples from their composition.

# References
