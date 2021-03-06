"""RGF (Regularized Greedy Forst) Classifier
http://stat.rutgers.edu/home/tzhang/software/rgf/
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto

Run the RGF model fit and predict for all the cross-validation folds
for each label number and model number specified.

Note: RGF currently only supports binary classificaiton. Thus, we run
the predictive simulations as many times as the number of labels
(9). Thus, these binary classification (training, prediction) can be
paralleliszed.

"""

import sys
import numpy as np
from sklearn import metrics

import rgf_utils as ru
from rgf_utils import mkdir_p


if len(sys.argv) < 4:
    print "python ./rgf_cv feature_number fold_number L2_value"
    sys.exit(1)

## cn is always 0
cn = int(sys.argv[1])
cn = 0 

fn = int(sys.argv[2])
L2 = float(sys.argv[3])
sL2 = L2/100.0

n_folds = 5
test_interval = 2000
max_leaf_forest = 10000

mn_max = int(max_leaf_forest / test_interval)

# directories
simdir = 'l2l3_' + sys.argv[3]
mkdir_p(simdir)
mkdir_p(simdir + '/data')
mkdir_p(simdir + '/input')
mkdir_p(simdir + '/output')
mkdir_p(simdir + '/model')

# print "cleaning up the data..."
train_csv = 'data/train.csv'
# ru.gen_l2l3files(train_csv, simdir=simdir, n_folds=n_folds)

# fit/predict for each cv fold
# training parameters
train_params = {'reg_L2': L2,
                'reg_sL2': sL2,
                'algorithm': 'RGF',
                'loss': 'Expo',  # LS|Expo|Log
                'test_interval': test_interval,
                'max_leaf_forest': max_leaf_forest,
                'train_x_fn': simdir + '/data/X_train-%d' % fn,
                'train_y_fn': simdir + '/data/y_train_%d-%d' % (cn, fn),
                'model_fn_prefix':
                simdir + '/model/otto_%d-%d.model' % (cn, fn)
            }
train_inp = simdir + '/input/train_%d-%d.inp' % (cn, fn)

# fit each label
print 'Fitting each label to the model...'
ru.rgf_fit(train_params, train_inp)

print 'Predicting each label to the model...'
# predict on different models
for mn in range(5, mn_max + 1, 5):
    # predicting parameters
    predict_params = {'test_x_fn': simdir + '/data/X_test-%d' % fn,
                      'model_fn': simdir + '/model/otto_%d-%d.model-%02d'
                      % (cn, fn, mn),
                      'prediction_fn': simdir + '/output/y_%d-%d.pred-%02d'
                      % (cn, fn, mn)}
    predict_inp = simdir + '/input/predict_%d-%d-%d.inp' % (cn, fn, mn)

    # predict each label
    ru.rgf_predict(predict_params, predict_inp)


ypredfile = simdir + '/output/y_%d-' + '%d' % fn + '.pred-%02d' % mn_max
y_pred = np.loadtxt(ypredfile %cn, dtype=float)
y_true = np.loadtxt(simdir+'/data/y_test_%d-%d' % (cn, fn), dtype=int)

y_pred_int = np.array([1 if l>0 else -1 for l in y_pred])

print metrics.confusion_matrix(y_true, y_pred_int)
