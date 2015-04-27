"""
AdaBoostClassifier
sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
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
if n_args<2:
	sys.exit(1)

try:
	n_estimators = int(sys.argv[1])      # 250
	learning_rate = float(sys.argv[2])   # 1.0
	algorithm = sys.argv[3]              # SAMME.R
except:
	print 'parameter specification wrong!'
	sys.exit(1)

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

# set up the model
params = {'n_estimators': n_estimators,
		  'learning_rate': learning_rate,
		  'algorithm': algorithm}
clf = ensemble.AdaBoostClassifier(n_estimators=params['n_estimators'],
								  learning_rate=params['learning_rate'],
								  algorithm=params['algorithm'])

# write parameters
logfile = 'log_adb/adb_cv_%d_%f_%s' %(params['n_estimators'],
									 params['learning_rate'],
									 params['algorithm'])
pname = logfile+'.txt'
xu.write_params(pname, clf.get_params())

# write results
fscores = open(logfile+'.csv', 'w')

# run CV
scores = cross_validation.cross_val_score(clf, train, labels,
										  cv=2, scoring=logloss_mc,
										  verbose=0)
print "scores: ", scores
fscores.write('%f %f\n' %(scores.mean(), scores.std()))
fscores.close()


