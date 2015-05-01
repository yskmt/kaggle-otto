"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import json

import xgboost as xgb


pardir = os.path.realpath('..')
if pardir not in sys.path:
    sys.path.append(pardir)


from otto_utils import calc_ll_from_proba, load_train_data


def load_xgb_train_data(train_csv, train_buf):

    X, y, encoder, scaler = load_train_data(train_csv)
    
    # save data for xgboost
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.save_binary(train_buf)

    return X, y, encoder, scaler, dtrain

    
def xgb_cv(params, dtrain, num_rounds, nfold=5):
    print 'cross validation started...'
    
    # cross-validatoin
    t0 = time.time()
    cv_result = xgb.cv(params, dtrain, num_rounds, nfold=5,
                       metrics={'mlogloss'}, seed=0)
    print "elapsed time: ", (time.time() - t0)

    ll_test = []
    ll_train = []
    for li in cv_result:
        lis = li.split('\t')
        ll_test.append(lis[1].split(':')[1].split('+')[0])
        ll_train.append(lis[2].split(':')[1].split('+')[0])

    ll_test = np.array(ll_test, dtype=float)
    ll_train = np.array(ll_train, dtype=float)
    lls = np.array([ll_test, ll_train]).T

    return lls


def write_params(pname, params):
    print 'writing the parameters to file...'
    
    with open(pname, 'w') as f:
        json.dump(params, f)


def plot_cv_results(fname):

    lls = np.loadtxt(fname)
    eval_ll = lls[:,0]
    train_ll = lls[:,1]

    plt.plot(eval_ll, label='eval-mlogloss')
    plt.plot(train_ll, label='train-mlogloss')
    plt.legend(loc='best')
    plt.show()

    return eval_ll, train_ll
