from __future__ import absolute_import
from __future__ import print_function

import json
import sys
import os

import numpy as np
import pandas as pd

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
from sklearn import cross_validation
from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

pardir = os.path.realpath('..')
if pardir not in sys.path:
    sys.path.append(pardir)

import otto_utils as ou


############################################################################


def softsign(x):
    return x / (1 + abs(x))


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
            model.save_weights("%s/weights-%d-%d-%d.hdf5" %
                               (simname, simnum, ncv, fi))
            log_train = model.log_train
            log_valid = model.log_validation
            # early stopping
            if log_valid[-1] > log_valid[-1 - epoch_step / 2]:
                break

        print("Predicting on test set...")
        proba = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
        probas.append(proba)

        ll.append(ou.calc_ll_from_proba(proba, y_test))

        np.savetxt('%s/proba-%d-%d.log' % (simname, simnum, ncv), proba)
        ncv += 1

    np.savetxt('%s/ll-%d.txt' % (simname, simnum), np.array(ll))
    return probas


def keras_bagging(simname, simnum, params, X, y,
                  n_folds=3, nb_epoch=1000, batch_size=256, vb=2):
    """Carry out the k-fold cross validation of the NN with given
    parameters with bagging.

    * Uses sklearn.ensemble.BaggingClassifier for bagging.
    * Uses keras_wrapper class to make it compatible with     
    * NOTE: dimensionality of y is different from what kearas uses.
    y is a vector with corresponding labels (0 to nb_classes-1) 
    

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
    lls = []
    ncv = 0
    for train_index, test_index in kf:
        print ("cross-validation: %dth fold..." % ncv)

        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        dims = X_train.shape[1]
        nb_classes = max(y_train)+1

        params['dims'] = dims
        params['nb_classes'] = nb_classes

        print(nb_classes, 'classes')
        print(dims, 'dims')
        print("Fitting the model on train set...")
        model = keras_wrapper(nb_epoch=nb_epoch, **params)

        bagg = BaggingClassifier(model, n_estimators=5,
                                 max_samples=0.5, bootstrap=True,
                                 bootstrap_features=False,
                                 oob_score=True, n_jobs=1,
                                 random_state=1234, verbose=1)
        bagg.fit(X_train, y_train)

        print("Predicting on test set...")
        proba = bagg.predict_proba(X_test)
        probas.append(proba)
        lls.append(ou.calc_ll_from_proba(proba, y_test))

        np.savetxt('%s/proba-%d-%d.log' % (simname, simnum, ncv), proba)
        ncv += 1

    np.savetxt('%s/ll-%d.txt' % (simname, simnum), np.array(lls))
    return probas, lls


class keras_wrapper(BaseEstimator, ClassifierMixin):
    """Wrapper function around keras to make it compatible with
    scikit-learn.

    """

    
    def __init__(self, layer_size, dropout_rate, nb_classes, dims,
                 prelu=False, batchnorm=False, opt='sgd', sgd_lr=0.01,
                 sgd_mom=0.9, sgd_decay=0.0, sgd_nesterov=True,
                 activation_func='tanh', weight_ini='glorot_uniform',
                 reg=[None, None], max_constraint=False,
                 input_dropout=0.0, nb_epoch=1000, batch_size=256,
                 validation_split=0.15, verbose=2):

        self.nb_classes = nb_classes
        self.dims = dims
        self.layer_size = layer_size
        self.opt = opt
        self.sgd_lr = sgd_lr
        self.sgd_decay = sgd_decay
        self.sgd_mom = sgd_mom
        self.sgd_nesterov = sgd_nesterov
        self.activation_func = activation_func
        self.weight_ini = weight_ini
        self.batchnorm = batchnorm
        self.prelu = prelu
        self.dropout_rate = dropout_rate
        self.reg = reg
        self.max_constraint = max_constraint
        self.input_dropout = input_dropout
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        
        self.build_keras_model(layer_size=layer_size,
                               dropout_rate=dropout_rate,
                               nb_classes=nb_classes,
                               dims=dims,
                               prelu=prelu,
                               batchnorm=batchnorm,
                               opt=opt,
                               sgd_lr=sgd_lr,
                               sgd_mom=sgd_mom,
                               sgd_decay=sgd_decay,
                               sgd_nesterov=sgd_nesterov,
                               activation_func=activation_func,
                               weight_ini=weight_ini,
                               reg=reg,
                               max_constraint=max_constraint,
                               input_dropout=input_dropout)

    def build_keras_model(self, layer_size, dropout_rate, nb_classes, dims,
                          prelu=False, batchnorm=False, opt='sgd',
                          sgd_lr=0.01, sgd_mom=0.9, sgd_decay=0.0,
                          sgd_nesterov=True, activation_func='tanh',
                          weight_ini='glorot_uniform',
                          reg=[None, None],
                          max_constraint=False, input_dropout=0.0):

        self.model = Sequential()
        self.model.early_stopping = 100
        nb_layers = len(layer_size)

        # initial dropout
        if input_dropout > 0.0:
            self.model.add(Dropout(input_dropout))

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

            self.model.add(Dense(in_layer_size, out_layer_size,
                                 init=weight_ini, activation=activation_func,
                                 W_regularizer=W_reg, b_regularizer=b_reg))

            if prelu is True:
                self.model.add(PReLU((out_layer_size,)))

            if batchnorm is True:
                self.model.add(BatchNormalization((out_layer_size,)))

            if dropout_rate[i] > 0.0:
                self.model.add(Dropout(dropout_rate[i]))

        # output layer
        self.model.add(Dense(layer_size[-1], nb_classes, init=weight_ini))
        self.model.add(Activation('softmax'))

        if opt is 'sgd':
            optimizer = keras.optimizers.SGD(
                lr=sgd_lr, momentum=sgd_mom, decay=sgd_decay,
                nesterov=sgd_nesterov)
        elif opt is 'adadelta':
            optimizer = Adadelta(lr=1.0, rho=0.99, epsilon=1e-8)
        else:
            optimizer = opt

        self.model.compile(
            loss='categorical_crossentropy', optimizer=optimizer)

        return self.model

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # transform y to categorical matrix form
        y = np_utils.to_categorical(y)
        
        self.model.fit(X, y,
                       nb_epoch=self.nb_epoch,
                       batch_size=self.batch_size,
                       validation_split=self.validation_split,
                       verbose=self.verbose)
        return self
        
    def predict_proba(self, X):
        return self.model.predict_proba(
            X, batch_size=self.batch_size, verbose=self.verbose)

    # def get_params(self, deep=True):
    #     return {"nb_classes": self.nb_classes,
    #             "dims": self.dims,
    #             "layer_size": self.layer_size,
    #             "opt": self.opt,
    #             "sgd_lr": self.sgd_lr,
    #             "sgd_decay": self.sgd_decay,
    #             "sgd_mom": self.sgd_mom,
    #             "sgd_nesterov": self.sgd_nesterov,
    #             "activation_func": self.activation_func,
    #             "weight_ini": self.weight_ini,
    #             "batchnorm": self.batchnorm,
    #             "prelu": self.prelu,
    #             "dropout_rate": self.dropout_rate,
    #             "reg": self.reg,
    #             "max_constraint": self.max_constraint,
    #             "input_dropout": self.input_dropout,
    #             "nb_epoch": self.nb_epoch,
    #             "batch_size": self.batch_size,
    #             "validation_split": self.validation_split,
    #             "verbose": self.verbose}

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self
