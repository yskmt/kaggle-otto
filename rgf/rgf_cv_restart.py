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
import rgf_utils as ru
from rgf_utils import mkdir_p

if len(sys.argv) < 4:
    print "python ./rgf_cv feature_number fold_number L2_value"
    sys.exit(1)

cn = int(sys.argv[1])
fn = int(sys.argv[2])
L2 = float(sys.argv[3])

n_folds = 5
test_interval   = 500
max_leaf_extra  = 10000
max_leaf_forest = 10000+max_leaf_extra

mn_max = int(max_leaf_extra/test_interval)

# directories
simdir = 'expo_' + sys.argv[3]
mkdir_p(simdir)
mkdir_p(simdir + '/data')
mkdir_p(simdir + '/input')
mkdir_p(simdir + '/output')
mkdir_p(simdir + '/model')
mkdir_p(simdir + '/input_2')
mkdir_p(simdir + '/output_2')
mkdir_p(simdir + '/model_2')


print "cleaning up the data..."
train_csv = 'data/train.csv'
# ru.gen_datafiles(train_csv, simdir=simdir, n_folds=n_folds)

# fit/predict for each cv fold
# training parameters
train_params = {'reg_L2': L2,
				'algorithm': 'RGF',
				'loss': 'Expo',  # LS|Expo|Log
				'test_interval': test_interval,
				'max_leaf_forest': max_leaf_forest,
				'train_x_fn': simdir + '/data/X_train-%d' % fn,
				'train_y_fn': simdir + '/data/y_train_%d-%d' % (cn, fn),
				'model_fn_prefix':
				simdir + '/model_2/otto_%d-%d.model' % (cn, fn),
				'model_fn_for_warmstart':
				simdir + '/model/otto_%d-%d.model-%d' %(cn, fn, 20)}
train_inp = simdir + '/input_2/train_%d-%d.inp' % (cn, fn)

# fit each label
ru.rgf_fit(train_params, train_inp)

# predict on different models
for mn in range(5, mn_max+1, 5): 
	# predicting parameters
	predict_params = {'test_x_fn': simdir + '/data/X_test-%d' % fn,
					  'model_fn': simdir + '/model_2/otto_%d-%d.model-%02d'
					  % (cn, fn, mn),
					  'prediction_fn': simdir + '/output_2/y_%d-%d.pred-%02d'
					  % (cn, fn, mn)}
	predict_inp = simdir + '/input_2/predict_%d-%d-%d.inp' % (cn, fn, mn)

	# predict each label
	ru.rgf_predict(predict_params, predict_inp)
