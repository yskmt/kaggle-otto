"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import matplotlib.pyplot as plt
import time
import json


# calculate the logloss score (smaller better)
def calculate_logloss(labels_proba, labels_test):

    # create a probability matrix for test labels (1s and 0s)
    N, m = labels_proba.shape
    test_proba = np.zeros((N, m))
    idx = np.array(list(enumerate(labels_test)))
    test_proba[idx[:, 0], idx[:, 1]] = 1

    logloss = -np.sum(
        test_proba * np.log(
            np.maximum(np.minimum(labels_proba, 1 - 1e-15), 1e-15))) / N

    return logloss

def clean_train(train_csv, train_buf, random=False):

    print 'cleaning data...'
    # import data
    train = pd.read_csv(train_csv)

    # drop ids and get labels
    labels = train.target.values
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)
    n, p = train.shape

    # shuffle the train data
    random_select = range(n)
    if random:
        np.random.shuffle(random_select)

    train_ = train.iloc[random_select]
    train = train_.as_matrix()
    labels_train = labels[random_select]

    # encode labels
    lbl_enc = preprocessing.LabelEncoder()
    labels_train = lbl_enc.fit_transform(labels_train)

    # save data for xgboost
    dtrain = xgb.DMatrix(train, label=labels_train)
    dtrain.save_binary(train_buf)

    return dtrain

    
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
