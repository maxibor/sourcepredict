#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from collections import Counter
from . import normalize

from . import utils


class sourceforest():

    def __init__(self, source, sink, labels):
        self.source = pd.read_csv(source, index_col=0)
        y = pd.read_csv(labels, index_col=0)
        self.y = y['labels']
        self.tmp_sink = pd.read_csv(sink, index_col=0, dtype='int64')
        self.combined = pd.DataFrame(pd.merge(
            left=self.source, right=self.tmp_sink, how='outer', left_index=True, right_index=True).fillna(0))
        return None

    def __repr__(self):
        return(f'A sourceforest object of source {self.source} and sink {self.tmp_sink}')

    def add_unknown(self, alpha):

        label_avg = int(np.average(list(dict(Counter(self.y)).values())))

        tmp_unk = self.tmp_sink.multiply(alpha).apply(np.floor)
        tmp_unk.columns = ["UNKNOWN_0"]
        unk_init = tmp_unk
        comb_unk = self.combined

        unk_labs = ["UNKNOWN_0"]

        for i in range(1, label_avg):
            unk_lab = f"UNKNOWN_{i}"
            unk_labs.append(unk_lab)
            tmp = unk_init.apply(lambda x: int(np.random.normal(x, 0.1*x)), 1)
            tmp = tmp.to_frame()
            tmp.columns = [unk_lab]
            tmp_unk = pd.merge(
                left=tmp_unk, right=tmp, how='outer', left_index=True, right_index=True).fillna(0)

        self.unk_labs = pd.Series(
            data=['unknown']*len(unk_labs), index=unk_labs)
        self.unk = tmp_unk

    def normalize(self, method):
        if method == 'RLE':
            self.normalized = normalize.RLE_normalize(self.combined)
        elif method == 'SUBSAMPLE':
            self.normalized = normalize.subsample_normalize_pd(self.combined)
        elif method == 'CLR':
            self.normalized = normalize.CLR_normalize(self.combined)
        self.normalized = pd.merge(left=self.normalized, right=self.unk,
                                   how='outer', left_index=True, right_index=True).fillna(0)
        self.feat = self.normalized.drop(self.tmp_sink.columns, axis=1).T
        self.feat = self.feat.loc[:,
                                  self.feat.columns[self.feat.quantile(0.8, 0) > 0]]
        self.sink = self.normalized.drop(
            self.source.columns, axis=1).drop(self.unk.columns, axis=1).T
        self.sink = self.sink.loc[:, self.feat.columns]
        self.y = self.y.append(self.unk_labs)

    def select_features(self, cv):
        clf = DecisionTreeClassifier()
        trans = RFECV(clf, cv=cv)
        d = self.feat
        X_trans = trans.fit_transform(d, self.y)
        columns_retained_RFECV = d.iloc[:,
                                        :].columns[trans.get_support()].values
        self.feat = self.feat.loc[:, columns_retained_RFECV]
        self.sink = self.sink.loc[:, columns_retained_RFECV]

    def rndForest(self, seed, threads, ratio, outfile, kfold):
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.feat, self.y, test_size=0.2, random_state=seed)

        rfc = RandomForestClassifier(random_state=seed, n_jobs=threads)

        param_grid = {
            'n_estimators': [500, 1000],
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }

        CV_rfc = GridSearchCV(
            estimator=rfc, param_grid=param_grid, cv=kfold, n_jobs=threads)
        print(
            f"Performing {kfold} fold cross validation on {threads} cores...")
        CV_rfc.fit(train_features, train_labels)

        rfc1 = RandomForestClassifier(
            random_state=seed, max_features=CV_rfc.best_params_['max_features'], n_estimators=CV_rfc.best_params_['n_estimators'], max_depth=CV_rfc.best_params_['max_depth'], criterion=CV_rfc.best_params_['criterion'], class_weight="balanced", n_jobs=threads)

        print(
            f"Training classifier with best parameters on {threads} cores...")
        rfc1.fit(train_features, train_labels)
        y_pred = rfc1.predict(test_features)
        print("Training Accuracy:", metrics.accuracy_score(
            test_labels, y_pred), "\n=================")
        self.sink_pred = rfc1.predict_proba(self.sink)
        utils.print_class(classes=rfc1.classes_, pred=self.sink_pred)
        utils.print_ratio(classes=rfc1.classes_,
                          pred=self.sink_pred, ratio_orga=ratio)
        utils.write_out(outfile=outfile, classes=rfc1.classes_,
                        pred=self.sink_pred)
