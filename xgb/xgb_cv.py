"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import sys
import os
import numpy as np
import json

import xgboost as xgb

import xgb_utils as xu


pardir = os.path.realpath('..')
if pardir not in sys.path:
    sys.path.append(pardir)

from otto_utils import (mkdir_p,
                        load_train_data, load_test_data, calc_ll_from_proba)


if len(sys.argv) < 10:
    print "python xgb_cv.py simdir simnum num_rounds eta gamma max_depth "\
        "min_child_weight colsample_bytree subsample"
    sys.exit(1)


# files
train_csv = '../data/train.csv'
train_buf = 'data/train.buffer'

# first clean the train data and save
if not os.path.isfile(train_buf):
    _, _, _, _, dtrain = xu.load_xgb_train_data(train_csv, train_buf)
else:
    dtrain = xgb.DMatrix(train_buf)

n_folds = 5

# set booster parameters
simdir = sys.argv[1]
simnum = int(sys.argv[2])
num_rounds = int(sys.argv[3])
cv_params = {
    "eval_metric": "mlogloss",
    "objective": "multi:softprob",
    "silent": 1,
    "num_class": 9,
    'nthread': 16,
    'eta': sys.argv[4],
    'gamma': sys.argv[5],
    'max_depth': sys.argv[6],
    'min_child_weight': sys.argv[7],
    'colsample_bytree': sys.argv[8],
    'subsample': sys.argv[9],
    'early_stopping_rounds': 10}


print cv_params

mkdir_p(simdir)
paramfile = simdir + '/param_%d.txt' % simnum
with open(paramfile, 'w') as fp:
    json.dump(cv_params, fp)

logfile = simdir + '/log_%d.txt' % simnum
lls = xu.xgb_cv(cv_params, dtrain, num_rounds, nfold=3)
np.savetxt(logfile, lls)
