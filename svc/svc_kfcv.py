"""
SupportVectorClassifier: K-fold cross validation with optimum parameters

optimum parameters:
{"simnum": 1, "C": 20.0, "simdir": "cv_4", "gamma": 0.0, "kernel": "rbf"}

sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""


import sys
import os
import errno
import json

import numpy as np

from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn.externals import joblib

from ..otto_utils import mkdir_p, calc_ll_from_proba


if len(sys.argv) < 5:
    sys.exit(1)

cv_params = {}
cv_params['simdir'] = sys.argv[1]
cv_params['kernel'] = sys.argv[2]
cv_params['C'] = float(sys.argv[3])
cv_params['gamma'] = float(sys.argv[4])

print "Loading train and label files..."
X = np.loadtxt('data/train_pped.csv')
y = np.loadtxt('data/labels_pped.csv')

n_folds = 5
n, p = X.shape

# create cv number of files for cross validation
kf = cross_validation.KFold(n, n_folds=n_folds,
                            shuffle=True,
                            random_state=1234)

ll = []
ncv = 0
for train_index, test_index in kf:
    print "cross-validation: %dth fold..." % ncv

    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    mkdir_p(cv_params['simdir'] + '/data')
    np.savetxt(cv_params['simdir'] + '/data/y_test-%d' % ncv, y_test)

    clf = svm.SVC(probability=True, verbose=False,
                  C=cv_params['C'], kernel=cv_params['kernel'],
                  gamma=cv_params['gamma'], cache_size=2000)
    print "fitting svc model.."
    clf.fit(X_train, y_train)

    print "predicting probabilities using svc model..."
    proba = clf.predict_proba(X_test)
    ll.append(calc_ll_from_proba(proba, y_test))

    print metrics.confusion_matrix(
        y_test.astype(int), np.argmax(proba, axis=1).astype(int))

    print "saving svc model..."
    joblib.dump(clf, cv_params['simdir'] + '/svc_%d.pkl' % ncv)

    ncv += 1

print ll
ll = np.array(ll)
ll_mean = ll.mean()
ll_std = ll.std()

np.savetxt(cv_params['simdir'] + '/res.txt', [ll_mean, ll_std])

pfile = cv_params['simdir'] + '/params.txt'
with open(pfile, 'w') as f:
    json.dump(cv_params, f)
