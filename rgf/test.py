import numpy as np

from sklearn import cross_validation
from sklearn import preprocessing

import rgf_utils as ru
from regularized_greedy_forest import RegularizedGreedyForestClassifier as RGFC

# simdir = 'expo_test'

# train_inp = simdir + '/input/train.inp'
# predict_inp = simdir + '/input/predict.inp'

# # firest generate data files
# # ru.gen_datafiles(simdir+'/data/train.csv', simdir, n_folds=5)

# fn = 0
# X_train = np.loadtxt('data/X_train-%d' % fn)
# y_train = np.loadtxt('data/y_train-%d' % fn)
# X_test = np.loadtxt('data/X_test-%d' % fn)
# y_true = np.loadtxt('data/y_test-%d' %fn)

# L2 = 0.1
# sL2 = L2 / 100


# rg = RGFC(simdir=simdir, algorithm="RGF", loss='Expo', reg_L2=L2, reg_sL2=sL2,
#           max_leaf_forest=100, test_interval=20, n_labels=9, n_jobs=-1)
# rg.fit(X_train, y_train)
# proba = rg.predict_proba(X_test)

# ll = ru.calc_ll_from_proba(proba, y_true)


##########################################################################
import pandas as pd


# # import data
# train_csv = 'data/train.csv'
# train = pd.read_csv(train_csv)

# # drop ids and get labels
# labels = train.target.values
# train = train.drop('id', axis=1)
# train = train.drop('target', axis=1)

# # change to numpy array
# train = train.values

# # encode labels
# lbl_enc = preprocessing.LabelEncoder()
# labels = lbl_enc.fit_transform(labels)

# # shuffle
# n, p = train.shape
# random_select = range(n)
# np.random.shuffle(random_select)
# train = train[random_select, :]
# labels = labels[random_select]

# np.savetxt('data/X_train.csv', train)
# np.savetxt('data/y_train.csv', labels)

n_folds = 5
n_labels = 9

X = np.loadtxt('data/X_train.csv')
y = np.loadtxt('data/y_train.csv')


cv_params = {}
cv_params['L2'] = 0.1
cv_params['sL2'] = 0.01
cv_params['algorithm'] = 'RGF'
cv_params['loss'] = 'Expo'
cv_params['max_leaf_forest'] = 10
cv_params['test_interval'] = 2
cv_params['simdir'] = 'expo_test'


def rgf_cv(X, y, n_folds, cv_params):

    L2 = cv_params['L2']
    sL2 = cv_params['sL2']
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
                  reg_L2=L2, reg_sL2=sL2,
                  max_leaf_forest=max_leaf_forest, test_interval=test_interval,
                  n_labels=n_labels, n_jobs=-1)
        rg.fit(X_train, y_train)
        proba = rg.predict_proba(X_test)
        ll.append(ru.calc_ll_from_proba(proba, y_test))

        ncv += 1

    ll = np.array(ll)
    return ll.mean(), ll.std()




ll_mean, ll_std = rgf_cv(X, y, n_folds, cv_params)
