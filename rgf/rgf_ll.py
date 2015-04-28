"""RGF (Regularized Greedy Forst) Classifier
http://stat.rutgers.edu/home/tzhang/software/rgf/ Otto Group product
classification challenge @ Kaggle

__author__ : Yusuke Sakamoto


Calculate the logloss score of each cross-validation fold.
Need to run this after all the cv simulations are complete.

"""
import sys
import os
import subprocess

import numpy as np
from sklearn import metrics

import rgf_utils as ru


call_rgf = ['perl', './call_exe.pl', './rgf1.2/bin/rgf']

if len(sys.argv) < 2:
    print "argument number wrong"
    sys.exit(1)

# simdir = 'expo2_0.005000%s' % sys.argv[1]
simdir = 'cv_%s' % sys.argv[1]
n_labels = 9
n_folds = 5
n_models = 12

lls_mean = []
lls_std = []

for mn in range(1, 12, 1):
    lls = []
    print "model #: ", mn
    for fn in range(n_folds):

        X_file = simdir + '/%d/data/X_test' % fn
        y_true = np.loadtxt(simdir + '/data/y_test-%d' % fn, dtype=float)
        label_preds = []

        for lbn in range(n_labels):
            model_file = simdir + '/%d/model/RGF-%d-%02d' % (fn, lbn, mn)
            predict_file = simdir + '/%d/output/y_pred-%d-%02d' % (fn, lbn, mn)

            predict_params = {'test_x_fn': X_file,
                              'model_fn': model_file,
                              'prediction_fn': predict_file}

            _predict_inp = simdir + \
                '/%d/input/predict-%d-%0d' % (fn, lbn, mn) + '.inp'

            if not os.path.isfile(predict_file):
                # write predict input file
                with open(_predict_inp, 'w') as pf:
                    for k in predict_params.keys():
                        pf.write('%s=%s\n' % (k, str(predict_params[k])))

                # RGF predict!
                subprocess.call(call_rgf + ['predict', _predict_inp[:-4]])

            label_preds.append(np.loadtxt(predict_file))

        label_preds = np.array(label_preds).T
        n, p = label_preds.shape

        proba = np.exp(label_preds) / \
            np.sum(np.exp(label_preds), axis=1)[:, None]

        print metrics.confusion_matrix(y_true.astype(int), np.argmax(proba, axis=1).astype(int))

        lls.append(ru.calc_ll_from_proba(proba, y_true))

    lls = np.array(lls)

    lls_mean.append(lls.mean())
    lls_std.append(lls.std())


print np.array([lls_mean, lls_std]).T
