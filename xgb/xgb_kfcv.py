"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import os
import sys
import numpy as np
import json

import xgboost as xgb

from sklearn import cross_validation
from sklearn import metrics

pardir = os.path.realpath('..')
if pardir not in sys.path:
    sys.path.append(pardir)


import xgb_utils as xu
import otto_utils as ou

if len(sys.argv)<2:
    sys.exit(1)

ncv = int(sys.argv[1])

if ncv not in [0, 1, 2, 3, 4]:
    sys.exit(1)

simdir = 'kfcv'
ou.mkdir_p(simdir)

simdir = simdir + '/' + str(ncv)
ou.mkdir_p(simdir)


num_rounds = 2000
params = """{"eval_metric": "mlogloss", "early_stopping_rounds": 10, "colsample_bytree": "0.5", "num_class": 9, "silent": 1, "nthread": 16, "min_child_weight": "4", "subsample": "0.8", "eta": "0.0125","objective": "multi:softprob", "max_depth": "14", "gamma": "0.025"}"""

params = json.loads(params)


# files
train_csv = '../data/train.csv'
train_buf = 'data/train.buffer'

# first clean the train data and save
print 'loading data...'
X, y, encoder, scaler = ou.load_train_data(train_csv)
n_folds = 5

n, p = X.shape

# create cv number of files for cross validation
kf = cross_validation.KFold(n, n_folds=n_folds,
                            shuffle=True,
                            random_state=1234)

ll = []
i = 0
for train_index, test_index in kf:
    if i != ncv:
        i += 1
        continue
    
    print "cross-validation: %dth fold..." % ncv

    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    np.savetxt(simdir + '/X-train-%d' % ncv, X_train)
    np.savetxt(simdir + '/X-test-%d' % ncv, X_test)
    np.savetxt(simdir + '/y-train-%d' % ncv, y_train)
    np.savetxt(simdir + '/y-test-%d' % ncv, y_test)

    print "fitting xgb model.."
    clf = xgb.train(params, xgb.DMatrix(X_train, label=y_train),
                    num_rounds)

    # save model
    print "saving xgb model..."
    clf.save_model(simdir + '/xgb-%d.model' % ncv)
    clf.dump_model(simdir + '/dump-%d.txt' % ncv)

    # predict
    print "predicting probabilities using xgb model..."
    proba = clf.predict(xgb.DMatrix(X_test))
    ll.append(ou.calc_ll_from_proba(proba, y_test))

    print metrics.confusion_matrix(
        y_test.astype(int), np.argmax(proba, axis=1).astype(int))

    i += 1

ll = np.array(ll)
print "logloss: ", ll

ll_mean = ll.mean()
ll_std = ll.std()

np.savetxt(simdir + '/res.txt', [ll_mean, ll_std])
