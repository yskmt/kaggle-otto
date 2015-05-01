"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import os
import numpy as np
import json

import xgboost as xgb

from sklearn import cross_validation
from sklearn import metrics

import xgb_utils as xu
import otto_utils as ou


simnum = 0

num_rounds = 10
params = {"subsample": 0.9,
          "nthread": 4,
          "eta": 0.0125,
          "gamma": 1,
          "colsample_bytree": 0.8,
          "max_depth": 14,
          "min_child_weight": 2,
          "objective": "multi:softprob",
          "eval_metric": "mlogloss",
          "num_class": 9,
          "silent": 1
          }


train_csv = '../data/train.csv'
ou.mkdir_p('data')
train_buf = 'data/train.buffer'


# first clean the train data and save
if not os.path.isfile(train_buf):
    X, y, encoder, scaler, dtrain\
        = xu.load_xgb_train_data('../data/train.csv', train_buf)
    # X_test, ids = ou.load_test_data('../data/test.csv', scaler)
else:
    dtrain = xgb.DMatrix(train_buf)

simdir = 'cv_%d' % simnum
fname = simdir
ou.mkdir_p(simdir)

with open(simdir + '/params.txt', 'w') as f:
    json.dump(params, f)

n_folds = 5
n = dtrain.num_row()

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

    # mkdir_p(cv_params['simdir'] + '/data')
    # np.savetxt(cv_params['simdir'] + '/data/y_test-%d' % ncv, y_test)

    print "fitting xgb model.."
    clf = xgb.train(params, xgb.DMatrix(X_train, label=y_train),
                    num_rounds)

    # save model
    print "saving xgb model..."
    clf.save_model(simdir + '/xgb.model')
    clf.dump_model(simdir + '/dump.raw.txt')

    # predict
    print "predicting probabilities using xgb model..."
    proba = clf.predict(xgb.DMatrix(X_test))
    ll.append(ou.calc_ll_from_proba(proba, y_test))

    print metrics.confusion_matrix(
        y_test.astype(int), np.argmax(proba, axis=1).astype(int))

    ncv += 1

ll = np.array(ll)
print "logloss: ", ll

ll_mean = ll.mean()
ll_std = ll.std()

np.savetxt(simdir + '/res.txt', [ll_mean, ll_std])
