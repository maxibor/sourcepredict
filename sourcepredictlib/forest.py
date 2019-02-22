#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from skbio.diversity import beta_diversity
from skbio import TreeNode
from ete3 import NCBITaxa
from io import StringIO
import umap
# import altair as alt

from collections import Counter
from . import normalize

from . import utils


class sourceforest():

    def __init__(self, source, sink, labels):
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

    def add_unknown(self, alpha):

        label_avg = int(np.average(list(dict(Counter(self.y)).values())))

        tmp_unk = self.tmp_sink.multiply(alpha).apply(np.floor)
        tmp_unk.columns = ["UNKNOWN_0"]
        unk_init = list(tmp_unk.loc[:, "UNKNOWN_0"])
        all_unk = [list(tmp_unk.loc[:, "UNKNOWN_0"])]
        unk_labs = ["UNKNOWN_0"]

        for i in range(1, label_avg):
            unk_lab = f"UNKNOWN_{i}"
            unk_labs.append(unk_lab)
            all_unk.append([int(np.random.normal(x, alpha*x))
                            for x in unk_init])

        self.unk = pd.DataFrame(data=all_unk).transpose()
        self.unk.columns = unk_labs
        self.unk.index = self.tmp_sink.index
        self.unk_labs = pd.Series(
            data=['unknown']*len(unk_labs), index=unk_labs)

    def normalize(self, method):
        if method == 'RLE':
            self.normalized = normalize.RLE_normalize(self.combined)
        elif method == 'SUBSAMPLE':
            self.normalized = normalize.subsample_normalize_pd(self.combined)
        elif method == 'CLR':
            self.normalized = normalize.CLR_normalize(self.combined)
        self.normalized_unk = pd.merge(left=self.normalized, right=self.unk,
                                       how='outer', left_index=True, right_index=True).fillna(0)
        # self.feat_unk = self.normalized.drop(self.tmp_sink.columns, axis=1).T
        self.sink = self.normalized.drop(
            self.source.columns, axis=1).T
        # self.sink = self.sink.loc[:, self.normalized.index]
        self.y_unk = self.y_unk.append(self.unk_labs)

    def select_features(self, cv, quantile, threads):
        clf = DecisionTreeClassifier()
        trans = RFECV(clf, cv=cv, n_jobs=threads)
        d = self.feat
        d = d.loc[:, d.columns[d.quantile(quantile, 0) > 0]]
        X_trans = trans.fit_transform(d, self.y)
        columns_retained_RFECV = d.iloc[:,
                                        :].columns[trans.get_support()].values
        print(f"{len(columns_retained_RFECV)} features retained (from {len(self.feat.columns)}) after feature engineering")
        self.feat = self.feat.loc[:, columns_retained_RFECV]
        self.feat['label'] = self.y
        self.sink = self.sink.loc[:, columns_retained_RFECV]

    def dim_reduction(self, ndim):
        pca = PCA(n_components=ndim)
        pcaed = pca.fit_transform(self.normalized_unk.T)
        self.pcaed = pd.DataFrame(pcaed, index=self.normalized_unk.columns, columns=[
                                  f"PC{i+1}" for i in range(pcaed.shape[1])])
        self.feat_unk = self.pcaed.drop(self.tmp_sink.columns, axis=0)
        self.feat_unk['label'] = self.y_unk
        self.sink_unk = self.pcaed.drop(self.source.columns, axis=0).drop(
            self.unk.columns, axis=0)

        # to_plot = self.pcaed.copy()
        # to_plot['name'] = self.pcaed.index
        # to_plot['label'] = self.y

        # mychart = alt.Chart(to_plot).mark_circle(size=60).encode(
        #     x='PC1',
        #     y='PC2',
        #     color='label',
        #     tooltip=['label', 'name']
        # ).interactive()

        # mychart.save('/Users/borry/Documents/GitHub/sourcepredict/pca.html')

    def rndForest(self, seed, threads, outfile, kfold):
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.feat_unk.drop('label', axis=1), self.feat_unk.loc[:, 'label'], test_size=0.2, random_state=seed)

        rfc = RandomForestClassifier(random_state=seed, n_jobs=threads)

        param_rf_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }

        CV_rfc = GridSearchCV(
            estimator=rfc, param_grid=param_rf_grid, cv=kfold, n_jobs=threads)

        print(
            f"\tPerforming {kfold} fold cross validation on {threads} cores...")
        CV_rfc.fit(train_features, train_labels)

        rfc1 = RandomForestClassifier(
            random_state=seed,
            max_features='auto',
            n_estimators=CV_rfc.best_params_['n_estimators'],
            max_depth=CV_rfc.best_params_['max_depth'],
            criterion=CV_rfc.best_params_['criterion'],
            class_weight="balanced",
            n_jobs=threads)

        print(
            f"\tTraining random forest classifier with best parameters on {threads} cores...")
        rfc1.fit(train_features, train_labels)
        y_pred = rfc1.predict(test_features)
        print("\t-> Testing Accuracy:", metrics.accuracy_score(
            test_labels, y_pred))
        self.sink_pred_unk = rfc1.predict_proba(self.sink_unk)
        predictions = utils.class2dict(
            classes=rfc1.classes_, pred=self.sink_pred_unk)
        print(
            f"\t----------------------\n\t- Unknown: {predictions['unknown']*100}%")
        return(predictions)

    def compute_distance(self, rank='species'):
        # Getting a single Taxonomic rank
        ncbi = NCBITaxa()
        only_rank = []
        for i in list(self.normalized.index):
            try:
                if ncbi.get_rank([i])[i] == rank:
                    only_rank.append(i)
            except KeyError:
                continue
        self.normalized_rank = self.normalized.loc[only_rank, :]
        tree = ncbi.get_topology(
            list(self.normalized_rank.index), intermediate_nodes=False)
        newick = TreeNode.read(StringIO(tree.write()))
        wu = beta_diversity("weighted_unifrac", self.normalized_rank.T.as_matrix().astype(int), ids=list(
            self.normalized_rank.columns), otu_ids=[str(i) for i in list(self.normalized_rank.index)], tree=newick)
        self.wu = wu.to_data_frame()

    def embed(self, umap_csv, n_comp=200):
        my_umap = umap.UMAP(metric='precomputed',
                            n_neighbors=30, min_dist=0.03, n_components=n_comp, n_epochs=500)
        umap_embed_a = my_umap.fit(self.wu)
        cols = [f"PC{i}" for i in range(1, n_comp+1)]
        self.umap = pd.DataFrame(
            umap_embed_a.embedding_, columns=cols, index=self.normalized_rank.columns)

        if umap_csv:
            to_write = self.umap.copy(deep=True)
            y = self.y.copy(deep=True)
            y = y.append(
                pd.Series(data=['sink']*len(list(self.tmp_sink.columns)), index=self.tmp_sink.columns))
            to_write['label'] = y
            to_write['name'] = to_write.index
            to_write.to_csv(umap_csv)

            # mychart = alt.Chart(to_write).mark_circle(size=60).encode(
            #     x='PC1',
            #     y='PC2',
            #     color='label',
            #     tooltip=['label', 'name']
            # ).interactive()

            # mychart.save(
            #     '/Users/borry/Documents/GitHub/sourcepredict/umap.html')

        self.feat = self.umap.drop(self.tmp_sink.columns, axis=0)
        self.feat['label'] = self.y
        self.sink = self.umap.drop(self.source.columns, axis=0)

    def knn_classification(self, kfold, threads, seed):
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.feat.drop('label', axis=1), self.feat.loc[:, 'label'], test_size=0.2, random_state=seed)
        knn = KNeighborsClassifier(n_jobs=threads)

        param_knn_grid = {
            'n_neighbors': [3, 5, 10, 15, 20],
            'weights': ['uniform', 'distance']
        }

        CV_knn = GridSearchCV(
            estimator=knn, param_grid=param_knn_grid, cv=kfold, n_jobs=threads)

        print(
            f"\tPerforming {kfold} fold cross validation on {threads} cores...")
        CV_knn.fit(train_features, train_labels)

        knn1 = KNeighborsClassifier(
            n_neighbors=CV_knn.best_params_['n_neighbors'], weights=CV_knn.best_params_['weights'], n_jobs=threads)

        knn1.fit(train_features, train_labels)
        y_pred = knn1.predict(test_features)
        print("\t-> Testing Accuracy:", metrics.accuracy_score(
            test_labels, y_pred))
        self.sink_pred = knn1.predict_proba(self.sink)
        utils.print_class(classes=knn1.classes_, pred=self.sink_pred)
        predictions = utils.class2dict(
            classes=knn1.classes_, pred=self.sink_pred)
        return(predictions)
