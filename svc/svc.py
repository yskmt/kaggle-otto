"""
SupportVectorClassifier
sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""


import sys
import os
import json

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

pardir = os.path.realpath('..')
if pardir not in sys.path:
    sys.path.append(pardir)

from otto_utils import mkdir_p, logloss_mc



if not os.path.isfile('data/train_pped.csv'):

    # import data
    print "reading data..."
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
    np.random.seed(1234)
    np.random.shuffle(random_select)
    train = train[random_select, :]
    labels = labels[random_select]

    scaler = preprocessing.StandardScaler().fit(train)
    train = scaler.transform(train)

    print "Saving preprocessed train and labels..."
    np.savetxt('data/train_pped.csv', train)
    np.savetxt('data/labels_pped.csv', labels)

print "Loading train and label files..."
train = np.loadtxt('data/train_pped.csv')
labels = np.loadtxt('data/labels_pped.csv')

n_cvs = 5

cv_params = {}
cv_params['simdir'] = sys.argv[1]
cv_params['simnum'] = int(sys.argv[2])
cv_params['kernel'] = sys.argv[3]
cv_params['C'] = float(sys.argv[4])
cv_params['gamma'] = float(sys.argv[5])


clf_svc = svm.SVC(probability=True, verbose=False,
                  C=cv_params['C'], kernel=cv_params['kernel'],
                  gamma=cv_params['gamma'], cache_size=1000)

clf = make_pipeline(preprocessing.StandardScaler(), clf_svc)

mkdir_p(cv_params['simdir'])

scores = cross_validation.cross_val_score(clf, train, labels,
                                          cv=n_cvs, n_jobs=-1,
                                          scoring=logloss_mc,
                                          verbose=1)
print "scores: ", scores

logfile = cv_params['simdir'] + '/%d.txt' % cv_params['simnum']
with open(logfile, 'w') as fscores:
    fscores.write('%f %f\n' % (scores.mean(), scores.std()))

pfile = logfile.replace('.txt', 'p.txt')
with open(pfile, 'w') as f:
    json.dump(cv_params, f)
