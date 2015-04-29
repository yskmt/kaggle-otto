"""
SupportVectorClassifier
sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""


import sys
import os
import errno
import json

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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


# # import data
# train_csv = 'data/train.csv'
# train = pd.read_csv(train_csv)

# # drop ids and get labels
# labels = train.target.values
# train = train.drop('id', axis=1)
# train = train.drop('target', axis=1)

# # change to numpy array
# train = train.values

# # encode labels
# lbl_enc = preprocessing.LabelEncoder()
# labels = lbl_enc.fit_transform(labels)

# # shuffle
# n, p = train.shape
# random_select = range(n)
# np.random.seed(1234)
# np.random.shuffle(random_select)
# train = train[random_select, :]
# labels = labels[random_select]


# print "Saving shuffled train and labels..."
# np.savetxt('data/train_shuffled.csv', train)
# np.savetxt('data/labels_shuffled.csv', labels)

print "Loading train and label files..."
train = np.loadtxt('data/train_shuffled.csv')
labels = np.loadtxt('data/labels_shuffled.csv')


n_cvs = 5

cv_params = {}
cv_params['simdir'] = sys.argv[1]
cv_params['simnum'] = int(sys.argv[2])
cv_params['kernel'] = sys.argv[3]
cv_params['C'] = float(sys.argv[4])
cv_params['gamma'] = float(sys.argv[5])


clf = svm.SVC(probability=True, verbose=False,
              C=cv_params['C'], kernel=cv_params['kernel'],
              gamma=cv_params['gamma'])
mkdir_p(cv_params['simdir'])

scores = cross_validation.cross_val_score(clf, train, labels,
										  cv=n_cvs, n_jobs=-1,
										  scoring=logloss_mc,
										  verbose=1)
print "scores: ", scores

logfile = cv_params['simdir'] + '/%d.txt' % cv_params['simnum']
with open(logfile, 'w') as fscores:
    fscores.write('%d %f %f\n' % (scores.mean(), scores.std()))

pfile = logfile.replace('.txt', 'p.txt')
with open(pfile, 'w') as f:
	json.dump(cv_params, f)

