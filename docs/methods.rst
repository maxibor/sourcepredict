Methods
=======

All samples are first normalized to correct for uneven sequencing depth
using `GMPR`_ (default). After normalization, Sourcepredict performs a
two steps prediction.

Prediction of unknown sources proportion
----------------------------------------

The unknown sources proportion is the proportion of OTUs in the test
sample which are not present in the training dataset.

| Let :math:`S` be a sample of size :math:`O` with :math:`O` OTUs from
  the test dataset :math:`D\_{test}`
| Let :math:`n` be the average number of samples per class in the
  training dataset.
| Let :math:`U_n` be the samples to add to the training dataset to
  account for the unknown source proportion in a test sample.

First a :math:`\alpha` proportion (default=:math:`0.1`) of each
:math:`o_i` OTU (with :math:`i\in[1,O]`) is added to the training
dataset for each :math:`U_j` samples (with :math:`j\in[1,n]`), such as
:math:`U_j(o_i) = \alpha\times S_(o_i)`

The :math:`U_n` samples are then merged as columns to the training
dataset (:math:`D_{train}`) to create a new training dataset denoted
:math:`D\_{train\ unknown}`

| To predict this unknown proportion, the dimension of the training
  dataset :math:`D\_{train\ unknown}` (samples in columns, OTUs as rows)
  is first reduced to 20 with the scikit-learn implementation of the
  PCA.
| This training dataset is further divided into three subsets: train
  (64%), test (20%), and validation (16%).
| The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm
  is then trained on the train subset, and the test accuracy is computed
  with the test subset.
| The trained KNN model is then corrected for probability estimation of
  unknown proportion using the scikit-learn implementation of the
  Platt’s scaling method with the validation subset. This procedure is
  repeated for each sample of the test dataset.

Prediction of known source proportion
-------------------------------------

First, only OTUs corresponding to the *species* taxonomic level are kept
using ETE toolkit. A distance matrix is then computed on the merged
training dataset :math:`D_{train}` and test dataset :math:`D_{test}`
using the scikit-bio implementation of weighted Unifrac distance
(default).

The distance matrix is then embedded in two dimensions using the
scikit-learn implementation of t-SNE.

The 2-dimensional embedding is then split back to training and testing
dataset.

| The training dataset is further divided into three subsets: train
  (64%), test (20%), and validation (16%).
| The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm
  is then trained on the train subset, and the test accuracy is computed
  with the test subset.
| The trained KNN model is then corrected for source proportion
  estimation using the scikit-learn implementation of the Platt’s method
  with the validation subset.

Combining unknown and source proportion
---------------------------------------

For each sample, the predicted unknown proportion :math:`p\_{unknown}`
is then combined with the predicted proportion of each of the :math:`C`
source class :math:`c` of the training dataset such as:

.. math:: \sum_{c=1}^{C} s_c + p_{unknown} = 1


with

.. math:: s_c = s_{c predicted} \times p_{unknown}

.. _GMPR: https://peerj.com/articles/4600/