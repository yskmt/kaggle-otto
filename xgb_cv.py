"""
Try xgboost
xgboost: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import sys
import os
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import xgb_utils as xu


# dafult parameters
# args = [eta, max_depth, subsamples, min_child_weight, gamma, colapse_bytree
params = {'max_depth': 10,
          'objective': 'multi:softprob',
          'num_class': 9,
          'min_child_weight': 4,
          'subsample': .9,
          'gamma': 1,
          'colsample_bytree': .8,
          'silent': 1,
          'nthread': 4}

# cross validation on eta
etas = [0.3, 0.1, 0.05, 0.01, 0.005, 0.025]
max_depths = [10, 8, 12, 14, 16]
subsamples = [0.9, 0.8, 0.7]
minc_weights = [4, 2, 1]
gammas = [1]
colb_trees = [0.8]

# input parameters
args = [0] * 6
n_args = len(sys.argv)-1
for i in range(n_args):
	args[i] = int(sys.argv[1+1])

train_csv = 'train.csv'
train_buf = 'train.buffer'

# first clean the train data and save
if not os.path.isfile(train_buf):
    dtrain = xu.clean_train(train_csv, train_buf)
else:
    dtrain = xgb.DMatrix(train_buf)

# booster parameters from christdubois
params = {'eta': etas[args[0]],
		  'max_depth': max_depths[args[1]],
          'objective': 'multi:softprob',
          'num_class': 9,
          'min_child_weight': minc_weights[args[3]],
          'subsample': subsamples[args[2]],
          'gamma': gammas[args[4]],
          'colsample_bytree': colb_trees[args[5]],
          'silent': 1,
          'nthread': 4}
num_rounds = 2

logdir = 'log/'
fname = logdir+'xg_cv_eta_%.4f_md_%d_ss_%.1f_mw_%d_g_%d_ct_%.1f'\
		%(params['eta'], params['max_depth'], params['subsample'],
		  params['min_child_weight'], params['gamma'], params['colsample_bytree'])

xu.write_params(fname+'.txt', params)
lls = xu.xgb_cv(params, dtrain, num_rounds, nfold=5)
np.savetxt(fname+'.csv', lls)

sys.exit(0)

# plt.plot(lls[0, :], label='test')
# plt.plot(lls[1, :], label='train')
# plt.legend(loc='best')
# plt.show()
