"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import os
import xgboost as xgb
import matplotlib.pyplot as plt
import xgb_utils as xu


train_csv = 'train.csv'
train_buf = 'train.buffer'

# first clean the train data and save
if not os.path.isfile(train_buf):
    dtrain = xu.clean_train(train_csv, train_buf)
else:
    dtrain = xgb.DMatrix(train_buf)

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

lls = xu.xgb_cv(params, dtrain, num_rounds, nfold=5)

plt.plot(lls[0, :], label='test')
plt.plot(lls[1, :], label='train')
plt.legend(loc='best')
plt.show()
