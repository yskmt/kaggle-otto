from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
from time import time

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import ParameterSampler
from sklearn import cross_validation


# np.random.seed(1337) # for reproducibility
np.random.seed(int(time()))  # for reproducibility


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
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(y, encoder=None, categorical=True):
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


print("Loading data...")
X, labels = load_data('../data/train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)


nb_classes = y.shape[1]

# X_test, ids = load_data('../data/test.csv', train=False)
# X_test, _ = preprocess_data(X_test, scaler)


# create cv number of files for cross validation
n_folds = 5
n_samples = X.shape[0]
kf = cross_validation.KFold(n_samples, n_folds=n_folds,
                            shuffle=True,
                            random_state=1234)


def softsign(x):
    return x / (1 + abs(x))


def build_keras_model(nb_layers, layer_size, dropout_rate, nb_classes,
                      prelu=False, batchnorm=False, opt='sgd',
                      sgd_lr=0.01, sgd_mom=0.9, sgd_decay=0.0,
                      sgd_nesterov=True, activation_func='tanh',
                      weight_ini='glorot_uniform', l2_reg=0.0):

    model = Sequential()
    model.early_stopping = 5
    
    # input layer
    if l2_reg > 0.0:
        model.add(Dense(dims, layer_size,
                        init=weight_ini, activation=activation_func))
    else:
        model.add(Dense(dims, layer_size,
                        init=weight_ini, activation=activation_func))

    # hidden layers
    for i in range(nb_layers):

        if l2_reg > 0.0:
            model.add(Dense(layer_size, layer_size,
                            init=weight_ini, activation=activation_func,
                            W_regularizer=l2(l2_reg)))
        else:
            model.add(Dense(layer_size, layer_size,
                            init=weight_ini, activation=activation_func))

        if prelu is True:
            model.add(PReLU((layer_size,)))

        if batchnorm is True:
            model.add(BatchNormalization((layer_size,)))

        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(layer_size, nb_classes, init=weight_ini))
    model.add(Activation('softmax'))

    if opt is 'sgd':
        optimizer = keras.optimizers.SGD(
            lr=sgd_lr, momentum=sgd_mom, decay=sgd_decay,
            nesterov=sgd_nesterov)
    elif opt is 'adam':
        optimizer = 'adam'

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


print("Training model...")


batch_size = 16
nb_epoch = 100

# param_dist = {
#     'nb_layers': [3],
#     'layer_size': [512],
#     'dropout_rate': [0.5],
#     # 'prelu': [True, False],
#     'prelu': [True],
#     # 'batchnorm': [True, False],
#     'batchnorm': [True],
#     'opt': ['sgd'],
#     'sgd_lr': [0.01],
#     'sgd_mom': [0.9],
#     'sgd_decay': [0.0],
#     # 'sgd_nesterov': [True, False],
#     'sgd_nesterov': [True],
#     # 'activation_func': ['tanh'],
#     'activation_func': [softsign],
#     'weight_ini': ['glorot_uniform'],
#     'l2_reg': [0.0],
#     'nb_classes': [nb_classes]
# }

param_dist = {
    'nb_layers': [2, 4, 6, 8],
    'layer_size': [64, 128, 256, 512, 1024],
    'dropout_rate': [0.0, 0.25, 0.5],
    'prelu': [True, False],
    # 'prelu': [False],
    'batchnorm': [True, False],
    # 'batchnorm': [False],
    'opt': ['sgd'],
    'sgd_lr': [0.01],
    'sgd_mom': [0.9],
    'sgd_decay': [0.0],
    # 'sgd_nesterov': [True, False],
    'sgd_nesterov': [True],
    'activation_func': ['tanh'],
    'weight_ini': ['glorot_uniform'],
    'l2_reg': [0.0],
    'nb_classes': [nb_classes]
}

nb_search = 240

param_list = list(ParameterSampler(param_dist, n_iter=nb_search))
param_dict = [dict((k, v) for (k, v) in d.items()) for d in param_list]

with open('log/params.txt', 'w') as f:
    f.write(str(param_dict))

for i in range(nb_search):
    print('%d-th iteration...' % i)

    f = open('log/params-%d.txt' % i, 'w')
    f.write(str(param_dict[i]))
    f.close()
    print(param_dict[i])

    ncv = 0
    ll = []
    first = 0
    for train_index, test_index in kf:
        print("cross-validation: %dth fold..." % ncv)

        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        nb_classes = y_train.shape[1]
        print(nb_classes, 'classes')

        dims = X_train.shape[1]
        print(dims, 'dims')

        print("Building model...")
        model = build_keras_model(**param_dict[i])

        model.fit(X_train, y_train, nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  validation_split=0.15)

        # proba = model.predict_proba(X_test)
        # ll.append(calc_ll_from_proba(proba, y_test))

        log_train = model.log_train
        log_valid = model.log_validation

        if first == 0:
            break

    print('minimum validation ll: %f at %d'
          % (np.min(log_valid), np.argmin(log_valid)))

    np.savetxt('log/ll-%d.txt' % i,
               np.array([model.log_train, model.log_validation]).T)
