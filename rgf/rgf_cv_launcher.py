import sys
import numpy as np

from sklearn import cross_validation

import rgf_utils as ru
from regularized_greedy_forest import RegularizedGreedyForestClassifier as RGFC


if len(sys.argv) < 9:
    sys.exit(1)

cv_params = {}
cv_params['simdir'] = sys.argv[1]
cv_params['algorithm'] = sys.argv[2]
cv_params['loss'] = sys.argv[3]
cv_params['L2'] = float(sys.argv[4])
cv_params['sL2'] = float(sys.argv[5])
cv_params['max_leaf_forest'] = int(sys.argv[6])
cv_params['test_interval'] = int(sys.argv[7])
cv_params['reg_depth'] = float(sys.argv[8])

n_folds = 5
n_labels = 9

X = np.loadtxt('data/X_train.csv')
y = np.loadtxt('data/y_train.csv')


def rgf_cv(X, y, n_folds, cv_params):

    L2 = cv_params['L2']
    sL2 = cv_params['sL2']
    reg_depth = cv_params['reg_depth']
    algorithm = cv_params['algorithm']
    loss = cv_params['loss']
    max_leaf_forest = cv_params['max_leaf_forest']
    test_interval = cv_params['test_interval']
    simdir = cv_params['simdir']

    n, p = X.shape

    # create cv number of files for cross validation
    kf = cross_validation.KFold(n, n_folds=n_folds,
                                shuffle=True,
                                random_state=1234)

    ll = []
    ncv = 0
    for train_index, test_index in kf:
        simdir_cv = simdir + '/' + str(ncv)
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        rg = RGFC(simdir=simdir_cv, algorithm=algorithm, loss=loss,
                  reg_L2=L2, reg_sL2=sL2, reg_depth=reg_depth,
                  max_leaf_forest=max_leaf_forest, test_interval=test_interval,
                  n_labels=n_labels, n_jobs=-1)
        rg.fit(X_train, y_train)
        proba = rg.predict_proba(X_test)
        ll.append(ru.calc_ll_from_proba(proba, y_test))

        ncv += 1

    ll = np.array(ll)
    return ll.mean(), ll.std()


ll_mean, ll_std = rgf_cv(X, y, n_folds, cv_params)
np.savetxt(cv_params['simdir']+'/res.txt', [ll_mean, ll_std])
