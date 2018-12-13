#!/usr/bin/env python -W ignore::DeprecationWarning

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize as sknormalize
from sklearn.model_selection import train_test_split
from sklearn import metrics
from . import normalize


class sources():

    def __init__(self, infile):
        self.file = pd.read_csv(infile, index_col=0)
        self.x = self.file.drop(['label'], axis=1)
        self.y = self.file.label
        self.sources = self.file[self.file['label'] != 'sink'].iloc[:, :-1]
        self.sink = self.file[self.file['label'] == 'sink'].iloc[:, :-1]

    def add_unknown(self, alpha):
        tmp = self.sink.multiply(alpha)
        tmp = tmp.apply(np.ceil)
        tmp = tmp.rename(index=lambda x: re.sub('.*', 'unknown', x))
        tmp = self.sources.append(tmp)
        self.train_x = tmp
        self.y = self.y.append(pd.Series(['Unknown'], index=['Unknown']))
        self.train_y = self.y.index[self.y != 'sink']

    def normalize(self):
        self.train_x = normalize.RLE_normalize(self.train_x)
        self.train_x = sknormalize(self.train_x)
        return(self.train_x)

    def rndForest(self, seed, threads):
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.train_x, self.train_y, test_size=0.1)
        self._forest = RandomForestClassifier(n_jobs=threads, n_estimators=50)
        self._forest.fit(train_features, train_labels)
        print(train_features)
        print(train_labels)
        y_pred = self._forest.predict(test_features)
        print("Accuracy:", metrics.accuracy_score(test_labels, y_pred))
        print(test_labels)
        return(y_pred)
