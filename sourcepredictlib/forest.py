#!/usr/bin/env python -W ignore::DeprecationWarning

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from . import normalize


class sourceforest():

    def __init__(self, source, sink):
        self.source = pd.read_csv(source, index_col=0)
        self.tmp_feat = self.source.drop(
            ['labels'], axis=0).apply(pd.to_numeric)
        self.y = self.source.loc['labels', :][1:]
        self.tmp_sink = pd.read_csv(sink, dtype='int64')
        self.combined = pd.DataFrame(pd.merge(
            left=self.tmp_feat, right=self.tmp_sink, how='outer', on='TAXID').drop(['TAXID'], axis=1).fillna(0))
        return None

    def normalize(self):
        print(type(self.combined))
        self.normalized = normalize.RLE_normalize(self.combined)
        self.feat = self.normalized.loc[:, self.source.columns[1:]].T
        self.sink = self.normalized.drop(self.source.columns[1:], axis=1).T
        return(self.feat, self.sink)

    def rndForest(self, seed, threads):
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.feat, self.y, test_size=0.2)
        self._forest = RandomForestClassifier(
            n_jobs=threads, n_estimators=1000)
        print("Training classifier")
        self._forest.fit(train_features, train_labels)
        y_pred = self._forest.predict(test_features)
        print("Training Accuracy:", metrics.accuracy_score(test_labels, y_pred))
        sink_pred = self._forest.predict_proba(self.sink)
        print(sink_pred)
