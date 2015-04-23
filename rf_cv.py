"""
RandomForestClassifier
sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import sys
import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import cross_validation
from sklearn import preprocessing

import xgb_utils as xu


def logloss_mc(model, x, y_true, epsilon=1e-15):
    """Multiclass logloss:
    https://github.com/ottogroup/kaggle/blob/master/benchmark.py

    Precit the probability of some model and calculate the logloss
    error for cross-validation.

    """

    # predict probability
    # print 'preciting probability for cv #: %d' % cv_count
    y_prob = model.predict_proba(x)

    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll

n_args = len(sys.argv)
if n_args < 2:
    sys.exit(1)

# arg1 = max_depth
arg1 = int(sys.argv[1])


# import data
train_csv = 'data/train.csv'
train = pd.read_csv(train_csv)

# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)

# change to numpy array
train = train.values

# encode labels
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# shuffle
n, p = train.shape
random_select = range(n)
np.random.shuffle(random_select)
train = train[random_select, :]
labels = labels[random_select]

params = {'n_estimators': 1000, 'max_features': 'auto', 'max_depth': arg1,
          'bootstrap': True, 'n_jobs': -1, 'verbose': 0}

clf = ensemble.RandomForestClassifier(n_jobs=params['n_jobs'],
                                      n_estimators=params['n_estimators'],
                                      max_depth=params['max_depth'],
                                      max_features=params['max_features'],
                                      verbose=params['verbose'],
                                      warm_start=True)
# oob_score=True)

print clf.get_params()

fscores = open('log_rf/scores_%d.csv' % arg1, 'a+', 1)  # line buffering
for i in range(1, params['n_estimators'] + 1):
    clf.set_params(n_estimators=i)
    scores = cross_validation.cross_val_score(clf, train, labels,
                                              cv=2, scoring=logloss_mc,
                                              verbose=0)
    print "CV #: %d" % i, "scores: ", scores
    fscores.write('%d %f %f\n' % (i, scores.mean(), scores.std()))

fscores.close()

# scores = cross_validation.cross_val_score(clf, train, labels, cv=5,
# scoring=logloss_mc, verbose=1)

# np.savetxt('log_rf/scores.csv', [scores.mean(), scores.std()])

fname = 'log_rf/params.txt'
xu.write_params(fname + '.txt', clf.get_params())
