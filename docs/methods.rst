Methods
=======

Starting with a numerical organism count matrix (samples as columns,
organisms as rows, obtained by a taxonomic classifier) of merged
references and sinks datasets, samples are first normalized relative to
each other, to correct for uneven sequencing depth using the GMPR_ method
(default). After normalization, Sourcepredict performs a
two-step prediction algorithm. First, it predicts the proportion of
unknown sources, *i.e.* which are not represented in the reference
dataset. Second it predicts the proportion of each known source of the
reference dataset in the sink samples.

Organisms are represented by their taxonomic identifiers (TAXID).

Prediction of unknown sources proportion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Let :math:`S_i \in \{S_1, .., S_n\}` be a sample from the normalized
  sinks dataset :math:`D_{sink}`,
  :math:`o_{j}^{\ i} \in \{o_{1}^{\ i},.., o_{n_o^{\ i}}^{\ i}\}` be an
  organism in :math:`S_i`, and :math:`n_o^{\ i}` be the total number of
  organisms in :math:`S_i`, with :math:`o_{j}^{\ i} \in \mathbb{Z}+`.
| Let :math:`m` be the mean number of samples per class in the reference
  dataset, such that :math:`m = \frac{1}{O}\sum_{i=1}^{O}S_i`.
| For each :math:`S_i` sample, I define :math:`||m||` estimated samples
  :math:`U_k^{S_i} \in \{U_1^{S_i}, ..,U_{||m||}^{S_i}\}` to add to the
  reference dataset to account for the unknown source proportion in a
  test sample.

Separately for each :math:`S_i`, a proportion denoted
:math:`\alpha \in [0,1]` (default = :math:`0.1`) of each of the
:math:`o_{j}^{\ i}` organism of :math:`S_i` is added to each
:math:`U_k^{S_i}` samples such that
:math:`U_k^{S_i}(o_j^{\ i}) = \alpha \cdot x_{i \ j}` , where
:math:`x_{i \ j}` is sampled from a Gaussian distribution
:math:`\mathcal{N}\big(S_i(o_j^{\ i}), 0.01)`.

The :math:`||m||` :math:`U_k^{S_i}` samples are then added to the
reference dataset :math:`D_{ref}`, and labeled as *unknown*, to create a
new reference dataset denoted :math:`{}^{unk}D_{ref}`.

| To predict the proportion of unknown sources, a Bray-Curtis_ pairwise dissimilarity matrix of all :math:`S_i` and
  :math:`U_k^{S_i}` samples is computed using scikit-bio. This distance
  matrix is then embedded in two dimensions (default) with the
  scikit-bio implementation of PCoA.
| This sample embedding is divided into three subsets:
  :math:`{}^{unk}D_{train}` (:math:`64\%`), :math:`{}^{unk}D_{test}`
  (:math:`20\%`), and :math:`{}^{unk}D_{validation}`\ (:math:`16\%`).

| The scikit-learn implementation of KNN algorithm is then trained on
  :math:`{}^{unk}D_{train}`, and the training accuracy is computed with
  :math:`{}^{unk}D_{test}`.
| This trained KNN model is then corrected for probability estimation of
  the unknown proportion using the scikit-learn implementation of
  Platt_’s scaling method with :math:`{}^{unk}D_{validation}`.

The proportion of unknown sources in :math:`S_i`, :math:`p_u \in [0,1]`
is then estimated using this trained and corrected KNN model.

Ultimately, this process is repeated independantly for each sink sample
:math:`S_i` of :math:`D_{sink}`.

Prediction of known source proportion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, only organism TAXIDs corresponding to the species taxonomic level
are retained using the ETE toolkit_. A weighted Unifrac (default)
pairwise distance_ matrix is then computed on the merged and
normalized training dataset :math:`D_{ref}` and test dataset
:math:`D_{sink}` with scikit-bio.

This distance matrix is then embedded in two dimensions (default) using
the scikit-learn implementation of t-SNE_.

The 2-dimensional embedding is then split back to training
:math:`{}^{tsne}D_{ref}` and testing dataset :math:`{}^{tsne}D_{sink}`.

| The training dataset :math:`{}^{tsne}D_{ref}` is further divided into
  three subsets: :math:`{}^{tsne}D_{train}` (:math:`64\%`),
  :math:`{}^{tsne}D_{test}` (:math:`20\%`), and
  :math:`{}^{tsne}D_{validation}` (:math:`16\%`).
| The KNN algorithm is then trained on the train subset, with a five
  (default) cross validation to look for the optimum number of
  K-neighbors. The training accuracy is then computed with
  :math:`{}^{tsne}D_{test}`. Finally, this second trained KNN model is
  also corrected for source proportion estimation using the scikit-learn
  implementation of the Platt’s method with
  :math:`{}^{tsne}D_{validation}`.

The proportion :math:`p_{c_s} \in [0,1]` of each of the :math:`n_s`
sources :math:`c_s \in \{c_{1},\ ..,\ c_{n_s}\}` in each sample
:math:`S_i` is then estimated using this second trained and corrected
KNN model.

Combining unknown and source proportion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then for each sample :math:`S_i` of the test dataset :math:`D_{sink}`,
the predicted unknown proportion :math:`p_{u}` is then combined with the
predicted proportion :math:`p_{c_s}` for each of the :math:`n_s` sources
:math:`c_s` of the training dataset such that
:math:`\sum_{c_s=1}^{n_s} s_c + p_u = 1` where
:math:`s_c = p_{c_s} \cdot p_u`.

Finally, a summary table gathering the estimated sources proportions is
returned as a ``csv`` file, as well as the t-SNE embedding sample
coordinates.

.. _GMPR: https://peerj.com/articles/4600/
.. _Bray-Curtis: https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1942268
.. _scikit-learn: https://scikit-learn.org/stable/
.. _method: http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639
.. _toolkit: http://etetoolkit.org/
.. _distance: https://www.ncbi.nlm.nih.gov/pubmed/17220268
.. _t-SNE: http://www.jmlr.org/papers/v9/vandermaaten08a.html
.. _Platt: http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639