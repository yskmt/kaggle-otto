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


def calc_ll_from_proba(label_proba, label_true, eps=1e-15):
    """
    Calculate the logloss from the probability matrix.
    """

    # create a probability matrix for test labels (1s and 0s)
    N, m = label_proba.shape

    logloss = -np.sum(
        label_true * np.log(
            np.maximum(np.minimum(label_proba, 1 - 1e-15), 1e-15))) / N

    return logloss


def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)  # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


def preprocess_data(X, scaler=None):
    if not scaler:
        # standard transformation (mean=0, std=1)
        scaler = StandardScaler()
        scaler.fit(X)

        # min/max transforamtion (min=-1, max=1)
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # scaler.fit(X)

        # # PCA
        # scaler = PCA()
        # scaler.fit(X)

    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

def softsign(x):
    return x / (1 + abs(x))


def build_keras_model(layer_size, dropout_rate, nb_classes, dims,
                      prelu=False, batchnorm=False, opt='sgd',
                      sgd_lr=0.01, sgd_mom=0.9, sgd_decay=0.0,
                      sgd_nesterov=True, activation_func='tanh',
                      weight_ini='glorot_uniform',
                      reg=[None, None],
                      max_constraint=False, input_dropout=0.0):

    model = Sequential()
    model.early_stopping = 100
    nb_layers = len(layer_size)

    # initial dropout
    if input_dropout > 0.0:
        model.add(Dropout(input_dropout))

    # layers
    if reg[0] is not None:
        W_reg = l2(reg[0])
        b_reg = l2(reg[1])
    else:
        W_reg = None
        b_reg = None
        
    for i in range(nb_layers):

        in_layer_size = (dims if i == 0 else layer_size[i - 1])
        out_layer_size = layer_size[i]

        model.add(Dense(in_layer_size, out_layer_size,
                        init=weight_ini, activation=activation_func,
                        W_regularizer=W_reg, b_regularizer=b_reg))

        if prelu is True:
            model.add(PReLU((out_layer_size,)))

        if batchnorm is True:
            model.add(BatchNormalization((out_layer_size,)))

        if dropout_rate[i] > 0.0:
            model.add(Dropout(dropout_rate[i]))

    # output layer
    model.add(Dense(layer_size[-1], nb_classes, init=weight_ini))
    model.add(Activation('softmax'))

    if opt is 'sgd':
        optimizer = keras.optimizers.SGD(
            lr=sgd_lr, momentum=sgd_mom, decay=sgd_decay,
            nesterov=sgd_nesterov)
    elif opt is 'adadelta':
        optimizer = Adadelta(lr=1.0, rho=0.99, epsilon=1e-8)
    else:
        optimizer = opt

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model



def keras_cv(simname, simnum, params, X, y,
             n_folds=3, nb_epoch=1000, batch_size=256, vb=2):
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

        # save weights for every 100 iterations
        epoch_step = 100
        for fi in range(0, nb_epoch, epoch_step):
            model.fit(X_train, y_train,
                      nb_epoch=epoch_step,
                      batch_size=batch_size,
                      validation_split=0.1, verbose=vb)
            model.save_weights("%s/weights-%d-%d-%d.hdf5" %(simname, simnum, ncv, fi))
            
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

