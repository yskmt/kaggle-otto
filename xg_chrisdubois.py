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

def clean_data():

    print 'cleaning data...'
    # import data
    train = pd.read_csv('train.csv')
    sample = pd.read_csv('sampleSubmission.csv')

    # drop ids and get labels
    labels = train.target.values
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)
    n, p = train.shape

    # shuffle the train data
    random_select = range(n)
    np.random.shuffle(random_select)

    train_ = train.iloc[random_select]
    train = train_.as_matrix()
    labels_train = labels[random_select]

    # encode labels
    lbl_enc = preprocessing.LabelEncoder()
    labels_train = lbl_enc.fit_transform(labels_train)

    # save data for xgboost
    dtrain = xgb.DMatrix(train, label=labels_train)
    dtrain.save_binary("train.buffer")


# first clean the train data and save
if not os.path.isfile('train.buffer'):
    clean_data()

dtrain = xgb.DMatrix('train.buffer')

# booster parameters from christdubois
params = {'max_depth': 10,
          'objective': 'multi:softprob',
          'num_class': 9,
          'min_child_weight': 4,
          'subsample': .9,
          'gamma': 1,
          'colsample_bytree': .8,
          'silent': 1,
          'nthread': 4}
num_rounds = 250

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

plt.plot(ll_test, label='test')
plt.plot(ll_train, label='train')
plt.legend(loc='best')
plt.show()
