Methods
=======

All samples are first normalized to correct for uneven sequencing depth
using GMPR_ (default). After normalization, Sourcepredict
performs a two steps prediction: first a prediction of the proportion of
unknown sources, i.e. not represented in the reference dataset. Then a
prediction of the proportion of each known source of the reference
dataset in the test samples.

Organism are represented by their taxonomic identifiers (TAXID).

Prediction of unknown sources proportion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Let :math:`S` be a sample of size :math:`O` organims from the test
  dataset :math:`D_{sink}`
| Let :math:`n` be the average number of samples per class in the
  reference dataset.
| I define :math:`U_n` samples to add to the training dataset to account
  for the unknown source proportion in a test sample.

To compute :math:`U_n`, a :math:`\alpha` proportion (default =
:math:`0.1`) of each :math:`o_i` organism (with :math:`i\in[1,O]`) is
added to the training dataset for each :math:`U_j` samples (with
:math:`j\in[1,n]`), such as :math:`U_j(o_i) = \alpha\times S_(o_i)`

The :math:`U_n` samples are then merged as columns to the reference
dataset (:math:`D_{ref}`) to create a new reference dataset denoted
:math:`D_{ref\ unknown}`

| To predict this unknown proportion, the dimension of the reference
  dataset :math:`D_{ref\ unknown}` (samples in columns, organisms as
  rows) is first reduced to 20 with the scikit-learn_
  implementation of PCA.
| This reference dataset is then divided into three subsets:
  :math:`D_{train\ unknown}` (64%), :math:`D_{test\ unknown}` (20%), and
  :math:`D_{validation unknown}`\ (16%).

| The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm
  is then trained on :math:`D_{train\ unknown}`, and the test accuracy
  is computed with :math:`D_{test\ unknown}` .
| The trained KNN model is then corrected for probability estimation of
  unknown proportion using the scikit-learn implementation of the
  Platt’s scaling method_ with :math:`D_{validation\ unknown}`.
  This procedure is repeated for each sample of the test dataset.

The proportion of unknown :math:`p_{unknown}` sources in each sample is
then computed using the trained and corrected KNN model.

Prediction of known source proportion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, only organism TAXID corresponding to the *species* taxonomic
level are kept using ETE toolkit_. A distance matrix is then
computed on the merged training dataset :math:`D_{ref}` and test dataset
:math:`D_{sink}` using the scikit-bio implementation of weighted Unifrac
distance_ (default).

The distance matrix is then embedded in two dimensions using the
scikit-learn implementation of t-SNE_.

The 2-dimensional embedding is then split back to training
:math:`D_{ref\ tsne}` and testing dataset :math:`D_{sink\ tsne}`.

| The training dataset :math:`D_{ref\ tsne}` is further divided into
  three subsets: :math:`D_{train\ tsne}` (64%), :math:`D_{test\ tsne}`
  (20%), and :math:`D_{validation\ tsne}` (16%).
| The scikit-learn implementation of K-Nearest-Neighbors (KNN) algorithm
  is then trained on the train subset, and the test accuracy is computed
  with :math:`D_{test\ tsne}`.
| The trained KNN model is then corrected for source proportion
  estimation using the scikit-learn implementation of the Platt’s method
  with :math:`D_{validation\ tsne}`.

The proportion of each source :math:`p_{c}` sources in each sample is
then computed using the trained and corrected KNN model.

.. _GMPR: https://peerj.com/articles/4600/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _method: http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639
.. _toolkit: http://etetoolkit.org/
.. _distance: https://www.ncbi.nlm.nih.gov/pubmed/17220268
.. _t-SNE: http://www.jmlr.org/papers/v9/vandermaaten08a.html