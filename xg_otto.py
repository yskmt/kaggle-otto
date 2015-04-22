"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
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
# Multiple evals can be handled in this way
plst += [('eval_metric', 'mlogloss')]
# plst += [('eval_metric', 'ams@0')]

# Specify validations set to watch performance
evallist = [(dtest, 'eval'), (dtrain, 'train')]
# evallist = [(dtest, 'eval')]

# training
num_round = 2000
# bst = xgb.train(plst, dtrain, num_round, evallist)
# bst.save_model('0002.model')
# cross-validation
cv_result = xgb.cv(param, dtrain, num_round, nfold=5,
                   metrics={'mlogloss'}, seed=0)

# dump model
# bst.dump_model('dump.raw.txt')
# dump model with feature map
# bst.dump_model('dump.raw.txt','featmap.txt')

# predict
# ypred = bst.predict(xgb.DMatrix(test))

# logloss = calculate_logloss(ypred, labels_test)
# print "logloss score: ", logloss


# plot predicted target against test target
# plt.plot(labels_test, labels_pred, '.')
# plt.show()


# create submission file
# preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
# preds.to_csv('benchmark.csv', index_label='id')
