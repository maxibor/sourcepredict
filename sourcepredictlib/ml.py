#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa as skbio_mds
from skbio import TreeNode
from ete3 import NCBITaxa
from io import StringIO
import umap
import warnings
import sys

from collections import Counter
from . import normalize

from . import utils


class sourceunknown():

    def __init__(self, source, sink, labels):
        """
        Init of sourceunknown object
        Combines sink and source in one pd Dataframe
        Args:
            - source(str): training data csv file with OTUs at index, 
                Samples as columns
            - sink(str): test data csv file with OTUs at index, 
                Samples as columns
            - labels(str): labels csv file with Samples in first column, 
                class in 2nd column

        """
        self.source = pd.read_csv(source, index_col=0, dtype='int64')
        y = pd.read_csv(labels, index_col=0)
        self.y = y['labels']
        self.y_unk = pd.Series(data=['known']*len(list(self.y)), index=y.index)
        self.tmp_sink = sink
        self.combined = pd.DataFrame(pd.merge(
            left=self.source, right=self.tmp_sink, how='outer', left_index=True, right_index=True).fillna(0))
        return None

    def __repr__(self):
        return(f'A sourceforest object of source {self.source} and sink {self.tmp_sink}')

    def add_unknown(self, alpha, seed):
        """
        Create unknown Samples from test sample
        N Random samples are created with N being average of class counts
        For each random samples OTU, count is taken from nornal distrib with a 
        mean of test OTU count.
        Args:
            - alpha(float): proportion of each OTU count from test samples 
                to include in unknown sample
            - seed(int): seed for random number generator
        """

        np.random.seed = seed
        label_avg = int(np.average(list(dict(Counter(self.y)).values())))

        tmp_unk = self.tmp_sink.multiply(alpha).apply(np.floor)
        tmp_unk.columns = ["UNKNOWN_0"]
        unk_init = list(tmp_unk.loc[:, "UNKNOWN_0"])
        all_unk = [list(tmp_unk.loc[:, "UNKNOWN_0"])]
        unk_labs = ["UNKNOWN_0"]

        for i in range(1, label_avg):
            unk_lab = f"UNKNOWN_{i}"
            unk_labs.append(unk_lab)
            all_unk.append([int(np.random.normal(x, 0.1))
                            for x in unk_init])

        self.unk = pd.DataFrame(data=all_unk).transpose()
        self.unk.columns = unk_labs
        self.unk.index = self.tmp_sink.index
        self.unk_labs = pd.Series(
            data=['unknown']*len(unk_labs), index=unk_labs)

    def normalize(self, method, threads):
        """
        Performs normalization of the count data to balance coverage differences
        and missing OTUs
        Args:
            - method(str): normalization method
            - threads(int): number of threads for parallelization
        """
        if method == 'RLE':
            self.normalized = normalize.RLE_normalize(self.combined)
        elif method == 'SUBSAMPLE':
            self.normalized = normalize.subsample_normalize_pd(self.combined)
        elif method == 'CLR':
            self.normalized = normalize.CLR_normalize(self.combined)
        elif method == 'GMPR':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.normalized = normalize.GMPR_normalize(
                    self.combined, threads)
        self.normalized_unk = pd.merge(left=self.normalized, right=self.unk,
                                       how='outer', left_index=True, right_index=True).fillna(0)
        self.labels = pd.Series(['known'] * self.normalized.shape[1] +
                                ['unknown'] * self.unk.shape[1], index=self.normalized_unk.columns, name='labels')
        try:
            self.sink = self.normalized.drop(
                self.source.columns, axis=1).T
        except KeyError:
            print(f"ERROR: Test sample present in training dataset")
            sys.exit(1)
        self.y_unk = self.y_unk.append(self.unk_labs)

    def compute_distance(self, rank='species'):
        """
        Sample pairwise distance computation
        Args:
            - rank(str): Taxonomics rank to keep for filtering OTUs
        """

        # Getting a single Taxonomic rank
        ncbi = NCBITaxa()
        only_rank = []
        for i in list(self.normalized_unk.index):
            try:
                if ncbi.get_rank([i])[i] == rank:
                    only_rank.append(i)
            except KeyError:
                continue
        self.normalized_rank = self.normalized_unk.loc[only_rank, :].T
        self.skbio_wu = beta_diversity("braycurtis", counts=self.normalized_rank.as_matrix().astype(int), ids=list(
            self.normalized_rank.index))
        self.wu = self.skbio_wu.to_data_frame()

    def embed(self, out_csv, seed, n_comp=200):
        """
        Embedding of a distance matrix in lower dimensions
        Args:
            - out_csv(str): Path to file for writing out embedding coordinates
            - seed(int): seed for random number generator
            - n_comp(int): dimension of embedding
        """
        method = 'MDS'
        cols = [f"PC{i}" for i in range(1, n_comp+1)]

        embed = skbio_mds(
            self.skbio_wu, number_of_dimensions=n_comp, method='fsvd')
        my_embed = pd.DataFrame()
        for i in range(n_comp):
            my_embed[f"PC{i+1}"] = list(embed.samples.loc[:, f"PC{i+1}"])

        self.my_embed = my_embed
        self.my_embed.set_index(self.wu.index, inplace=True)

        self.source = self.my_embed.drop(self.tmp_sink.columns, axis=0)
        # self.source['labels'] =
        self.source = self.source.merge(
            self.labels.to_frame(), left_index=True, right_index=True)
        self.sink = self.my_embed.loc[self.tmp_sink.columns, :]

        if out_csv:
            to_write = self.my_embed.copy(deep=True)
            to_write['labels'] = self.y_unk
            to_write['name'] = to_write.index
            to_write.to_csv(out_csv)

    def ml(self, seed, threads):
        """
        KNN machine learning to predict unknown proportion
        Correction of predicted probabilies with Platt scaling from sklearn
        Training on 64% of data, validation on 16%, test on 20%
        Args:
            - seed(int) seed for random number generator
            - threads(int) number of threads for parallelization
        Returns:
            - predictions(dict): Probability/proportion of each class
        """
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.source.drop('labels', axis=1), self.source.loc[:, 'labels'], test_size=0.2, random_state=seed)
        train_features, validation_features, train_labels, validation_labels = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=seed)

        print(
            f"\tTraining KNN classifier on {threads} cores...")

        knn1 = KNeighborsClassifier(
            n_neighbors=len(self.unk_labs), weights='distance', n_jobs=threads)
        knn1.fit(train_features, train_labels)
        y_pred = knn1.predict(test_features)
        print("\t-> Testing Accuracy:", round(metrics.accuracy_score(
            test_labels, y_pred), 2))
        cal_knn = CalibratedClassifierCV(knn1, cv='prefit', method='sigmoid')
        cal_knn.fit(validation_features, validation_labels)

        self.sink_pred_unk = cal_knn.predict_proba(self.sink)
        sample = [''.join(list(self.tmp_sink.columns))]
        predictions = utils.class2dict(
            samples=sample, classes=cal_knn.classes_, pred=self.sink_pred_unk)
        utils.print_class(samples=sample,
                          classes=cal_knn.classes_, pred=self.sink_pred_unk)
        return(predictions)


class sourcemap():
    def __init__(self, source, sink, labels, norm_method, threads=4):
        '''
        Init of sourceumap object
        Combines sink and source in one pd Dataframe
        Args:
            - source(str): training data csv file with OTUs at index, 
                Samples as columns
            - sink(str): test data csv file with OTUs at index, 
                Samples as columns
            - labels(str): labels csv file with Samples in first column, 
                class in 2nd column
            - norm_method(str): normalization method
        '''
        self.train = pd.read_csv(source, index_col=0)
        self.test = pd.read_csv(sink, index_col=0)
        combined = self.train.merge(
            self.test, how='outer', left_index=True, right_index=True).fillna(0)
        if norm_method == 'RLE':
            self.normalized = normalize.RLE_normalize(combined).T
        elif norm_method == 'SUBSAMPLE':
            self.normalized = normalize.subsample_normalize_pd(combined).T
        elif norm_method == 'CLR':
            self.normalized = normalize.CLR_normalize(combined).T
        elif norm_method == 'GMPR':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.normalized = normalize.GMPR_normalize(combined, threads).T
        self.train_samples = list(self.train.columns)
        self.test_samples = list(self.test.columns)
        labels = pd.read_csv(labels, index_col=0)
        self.labels = labels.loc[self.train.columns, 'labels']

    def compute_distance(self, distance_method, rank='species'):
        """
        Sample pairwise distance computation
        Args:
            - distance_method(str): distance method used
            - rank(str): Taxonomics rank to keep for filtering OTUs
        """
        # Getting a single Taxonomic rank
        ncbi = NCBITaxa()
        only_rank = []
        for i in list(self.normalized.columns):
            try:
                if ncbi.get_rank([i])[i] == rank:
                    only_rank.append(i)
            except KeyError:
                continue
        self.normalized_rank = self.normalized.loc[:, only_rank]
        tree = ncbi.get_topology(
            list(self.normalized_rank.columns), intermediate_nodes=False)
        newick = TreeNode.read(StringIO(tree.write()))
        self.skbio_wu = beta_diversity(distance_method, self.normalized_rank.as_matrix().astype(int), ids=list(
            self.normalized_rank.index), otu_ids=[str(i) for i in list(self.normalized_rank.columns)], tree=newick)
        self.wu = self.skbio_wu.to_data_frame()

    def embed(self, method, out_csv, seed, n_comp=200):
        """
        Embedding of a distance matrix in lower dimensions
        Args:
            - method(str): method used for embedding
            - out_csv(str): Path to file for writing out embedding coordinates
            - seed(int): seed for random number generator
            - n_comp(int): dimension of embedding
        """
        cols = [f"PC{i}" for i in range(1, n_comp+1)]

        if method == 'UMAP':
            embed = umap.UMAP(metric='precomputed',
                              n_neighbors=30, min_dist=0.03, n_components=n_comp, random_state=seed, n_epochs=500)
            my_embed = embed.fit(self.wu)
        elif method == 'TSNE':
            embed = TSNE(metric='precomputed',
                         n_components=n_comp, random_state=seed)
            my_embed = embed.fit(np.matrix(self.wu))
        elif method == 'MDS':
            embed = skbio_mds(
                self.skbio_wu, number_of_dimensions=n_comp, method='fsvd')
            my_embed = pd.DataFrame()
            for i in range(n_comp):
                my_embed[f"PC{i+1}"] = list(embed.samples.loc[:, f"PC{i+1}"])
        else:
            print(f"Error, {method} embedding method not supported")
            sys.exit(1)

        if method in (['TSNE', 'UMAP']):
            self.my_embed = pd.DataFrame(
                my_embed.embedding_, columns=cols, index=self.wu.index)
        elif method == 'MDS':
            self.my_embed = my_embed
            self.my_embed.set_index(self.wu.index, inplace=True)

        if out_csv:
            to_write = self.my_embed.copy(deep=True)
            y = self.labels.copy(deep=True)
            y = y.append(
                pd.Series(data=['sink']*len(list(self.test.columns)), index=self.test.columns, name='labels'))
            to_write = to_write.merge(y, left_index=True, right_index=True)
            to_write['name'] = to_write.index
            to_write.to_csv(out_csv)

        self.source = self.my_embed.drop(self.test_samples, axis=0)
        self.source = self.source.merge(
            self.labels.to_frame(), left_index=True, right_index=True)
        self.sink = self.my_embed.drop(self.train_samples, axis=0)

    def knn_classification(self, kfold, threads, seed):
        """
        KNN machine learning to predict unknown proportion
        Correction of predicted probabilies with Platt scaling from sklearn
        Training on 64% of data, validation on 16%, test on 20%
        Args:
            - kfold
            - threads(int) number of threads for parallelization
            - seed(int) seed for random number generator
        Returns:
            - predictions(dict): Probability/proportion of each class
        """
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.source.drop('labels', axis=1), self.source.loc[:, 'labels'], test_size=0.2, random_state=seed)
        train_features, validation_features, train_labels, validation_labels = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=seed)

        knn = KNeighborsClassifier(n_jobs=threads)

        param_knn_grid = {
            'n_neighbors': [10, 20, 50, 100]
        }

        CV_knn = GridSearchCV(
            estimator=knn, param_grid=param_knn_grid, cv=kfold, n_jobs=threads)

        print(
            f"\tPerforming {kfold} fold cross validation on {threads} cores...")
        CV_knn.fit(train_features, train_labels)

        knn1 = KNeighborsClassifier(
            n_neighbors=CV_knn.best_params_['n_neighbors'], weights='distance', n_jobs=threads)

        print(
            f"\tTrained KNN classifier with {CV_knn.best_params_['n_neighbors']} neighbors")
        knn1.fit(train_features, train_labels)
        y_pred = knn1.predict(test_features)
        print("\t-> Testing Accuracy:", round(metrics.accuracy_score(
            test_labels, y_pred), 2))

        cal_knn = CalibratedClassifierCV(knn1, cv='prefit', method='sigmoid')
        cal_knn.fit(validation_features, validation_labels)

        self.sink_pred = cal_knn.predict_proba(self.sink)
        utils.print_class(samples=self.sink.index,
                          classes=knn1.classes_, pred=self.sink_pred)
        predictions = utils.class2dict(
            samples=self.sink.index, classes=knn1.classes_, pred=self.sink_pred)
        return(predictions)
