"""
Try Ensemble of NN with different parameters...

"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
from time import time
import json

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.optimizers import Adadelta

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import ParameterSampler
from sklearn import cross_validation
from sklearn.decomposition import PCA

from keras_utils import (load_data, build_keras_model,
                         preprocess_data, preprocess_labels, calc_ll_from_proba)

simname = 'ens'

np.random.seed(1234)  # for reproducibility

print("Loading data...")
X, labels = load_data('../data/train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

dims = X.shape[1]
nb_classes = y.shape[1]
nb_models = 4

# simulation parameters
batch_size = 256
nb_epoch = 2
n_folds = 3

params = {"nb_classes": 9, "dims": dims,
          "layer_size": [512, 512, 512, 512],
          "opt": "adagrad", "sgd_lr": 0.1, "sgd_decay": 0.1,
          "sgd_mom": 0.9, "sgd_nesterov": False,
          "activation_func": "relu",
          "weight_ini": "glorot_uniform",
          "batchnorm": True, "prelu": True,
          "dropout_rate": [0.4, 0.4, 0.4, 0.4],
          "input_dropout": 0.2,
          "reg": [1e-5, 1e-5],
          "max_constraint": False}


def keras_cv(simname, simnum, params, n_folds, X, y):
    """Carry out the k-fold cross validation of the NN with given
    parameters.

    """

    # number of samples, number of features
    n, p = X.shape

    # save parametr values
    f = open('%s/params-%d.txt' % (simname, simnum), 'w')
    json.dump(params, f)
    f.close()

    # create cv number of files for cross validation
    kf = cross_validation.KFold(n, n_folds=n_folds,
                                shuffle=True,
                                random_state=1234)

    probas = []
    ll = []
    ncv = 0
    for train_index, test_index in kf:
        print ("cross-validation: %dth fold..." % ncv)

        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index, :], y[test_index, :]

        dims = X_train.shape[1]
        nb_classes = y_train.shape[1]

        params['dims'] = dims
        params['nb_classes'] = nb_classes

        print(nb_classes, 'classes')
        print(dims, 'dims')
        print("Fitting the model on train set...")
        model = build_keras_model(**params)
        model.fit(X_train, y_train,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_split=0.0)

        log_train = model.log_train
        log_valid = model.log_validation

        print("Predicting on test set...")
        proba = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
        probas.append(proba)

        ll.append(calc_ll_from_proba(proba, y_test))

        np.savetxt('%s/proba-%d-%d.log' % (simname, simnum, ncv), proba)
        ncv += 1

    np.savetxt('%s/ll-%d.txt' % (simname, simnum), np.array(ll))
    return probas


# Create and fit NN four times
res_proba = []
for i in range(nb_models):
    # i: simnum
    res_proba.append(keras_cv('test', i, params, n_folds, X, y))

# number of samples, number of features
n, p = X.shape
    
# create cv number of files for cross validation
kf = cross_validation.KFold(n, n_folds=n_folds,
                            shuffle=True,
                            random_state=1234)

ncv = 0
ll_each = np.zeros((nb_models,1))
ll_ens = []
for train_index, test_index in kf:
    print ("cross-validation: %dth fold..." % ncv)

    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index, :], y[test_index, :]

    ens_proba = np.zeros(np.array(res_proba[0][ncv]).shape)

    # average the probabilities of each model
    for i in range(nb_models):
        ll = calc_ll_from_proba(res_proba[i][ncv], y_test)
        ll_each[i] += ll
        print (ll)

        ens_proba += np.array(res_proba[i][ncv])

    ens_proba /= nb_models
    ll_ens.append( calc_ll_from_proba(ens_proba, y_test))
    print(ll_ens[ncv])

    # save the ensamble resutls
    np.savetxt('%s/proba-ens-%d.log' % (simname, ncv), ens_proba)
    
    ncv += 1

np.savetxt('%s/ll-ens.txt' % (simname), np.array(ll_ens))
