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
import time


# INPUT
if int(len(sys.argv) < 4):
	print "Number of arguments wrong!"
	sys.exit(1)

arg0 = int(sys.argv[1])
arg1 = int(sys.argv[2])
arg2 = int(sys.argv[3])

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


# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')
n, p = train.shape

# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
# test = test.drop('id', axis=1)

n, p = train.shape
N = n / 2

# splitting the data by random select
random_select = range(n)
np.random.shuffle(random_select)

train_ = train.iloc[random_select[:N]]
test_ = train.loc[random_select[N:]]

train = train_.as_matrix()
test = test_.as_matrix()
labels_train = labels[random_select[:N]]
labels_test = labels[random_select[N:]]

# encode labels
lbl_enc = preprocessing.LabelEncoder()
labels_train = lbl_enc.fit_transform(labels_train)
labels_test = lbl_enc.fit_transform(labels_test)

# xgboost
dtrain = xgb.DMatrix(train, label=labels_train)
dtrain.save_binary("train.buffer")
dtest = xgb.DMatrix(test, label=labels_test)
dtest.save_binary("test.buffer")

# booster parameters
param = {'bst:max_depth': 2, 'bst:eta': 0.05, 'silent': 1,
         'objective': 'multi:softprob', 'num_class': p}
param['nthread'] = 4
plst = param.items()
plst += [('eval_metric', 'mlogloss')]

# Specify validations set to watch performance
evallist = [(dtest, 'eval'), (dtrain, 'train')]
# evallist = [(dtest, 'eval')]

# cross-validatoin
num_round = 4000
etas = [0.1, 0.05, 0.01, 0.005]
subsamples = [1.0, 0.75, 0.5]
max_depths = [2, 3, 4, 5]

print 'eta: ', etas[arg0]
print 'subsample: ', subsamples[arg1]
print 'max_depth: ', max_depths[arg2]

param['bst:eta'] = etas[arg0]
param['bst:subsample'] = subsamples[arg1]
param['bst:max_depth'] = max_depths[arg2]

t0 = time.time()
cv_result = xgb.cv(param, dtrain, num_round, nfold=5,
                   metrics={'mlogloss'}, seed=0)
print "elapsed time: ", (time.time() - t0)
