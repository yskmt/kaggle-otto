"""RGF (Regularized Greedy Forst) Classifier
http://stat.rutgers.edu/home/tzhang/software/rgf/ Otto Group product
classification challenge @ Kaggle

__author__ : Yusuke Sakamoto


Calculate the logloss score of each cross-validation fold.
Need to run this after all the cv simulations are complete.

"""
import sys
import numpy as np
from sklearn import metrics

import rgf_utils as ru

if len(sys.argv) < 2:
    print "argument number wrong"
    sys.exit(1)

# simdir = 'expo2_0.005000%s' % sys.argv[1]
simdir = 'expo2_0.010000%s' % sys.argv[1]
n_labels = 9
n_folds = 5


lls_mean = []
lls_std = []

for mn in range(5, 41, 5):
    lls = []
    for fn in range(n_folds):

        ypredfile = simdir + '/output/y_%d-' + '%d' % fn + '.pred-%02d' % mn

        # true value
        y_true = np.loadtxt('data/y_test-%d' % fn, dtype=int)
        y_predict, _ = ru.construct_ens(n_labels, ypredfile)

        # ll = ru.calc_ll(y_true, y_predict)
        proba = ru.calc_proba(n_labels, ypredfile)
        ll = ru.calc_ll_from_proba(proba, y_true)

        print 'mlogloss: ', ll
        lls.append(ll)

        # confusion matrix plot:
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        print 'confusion matrix: '
        print metrics.confusion_matrix(y_true, y_predict)
        
    lls_mean.append(np.array(lls).mean())
    lls_std.append(np.array(lls).std())

print np.array([lls_mean, lls_std]).T

