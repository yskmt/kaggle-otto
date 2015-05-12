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
from keras.constraints import maxnorm
from keras.optimizers import Adadelta

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import ParameterSampler
from sklearn import cross_validation
from sklearn.decomposition import PCA

from keras_utils import load_data, build_keras_model, preprocess_data, preprocess_labels

simdir = 'dg8'

np.random.seed(1234)  # for reproducibility

print("Loading data...")
X, labels = load_data('../data/train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

dims = X.shape[1]
nb_classes = y.shape[1]

# simulation parameters
batch_size = 256
nb_epoch = 1000
nb_search = 18

# param_dist = {"nb_classes": [9], "dims": [dims],
#               "layer_size": [[1024, 512, 256],
#                              [512, 512, 512],
#                              [1024, 1024, 1024]],
#               "opt": ["adagrad"], "sgd_lr": [0.1], "sgd_decay": [0.1],
#               "sgd_mom": [0.9], "sgd_nesterov": [False],
#               "activation_func": ["relu"],
#               "weight_ini": ["glorot_uniform"],
#               "batchnorm": [False], "prelu": [False],
#               "dropout_rate": [[0.6, 0.6, 0.6], [0.5, 0.5, 0.5], [0.4, 0.4, 0.4]],
#               "input_dropout": [0.1, 0.2, 0.3],
#               "W_reg": [l2(1e-5)], "b_reg": [l2(1e-5)],
#               "max_constraint": [False]}

# param_dist = {"nb_classes": [9], "dims": [dims],
#               "layer_size": [[1024, 1024, 1024]],
#               "opt": ["adagrad"], "sgd_lr": [0.1], "sgd_decay": [0.1],
#               "sgd_mom": [0.9], "sgd_nesterov": [False],
#               "activation_func": ["relu"],
#               "weight_ini": ["glorot_uniform"],
#               "batchnorm": [True, False], "prelu": [True, False],
#               "dropout_rate": [[0.4, 0.4, 0.4]],
#               "input_dropout": [0.1, 0.2],
#               "W_reg": [l2(1e-5)], "b_reg": [l2(1e-5)],
#               "max_constraint": [False]}

param_dist = {"nb_classes": [9], "dims": [dims],
              "layer_size": [[1024, 1024, 1024, 1024],
                             [2048, 1024, 512, 256]],
              "opt": ["adagrad"], "sgd_lr": [0.1], "sgd_decay": [0.1],
              "sgd_mom": [0.9], "sgd_nesterov": [False],
              "activation_func": ["relu"],
              "weight_ini": ["glorot_uniform"],
              "batchnorm": [True], "prelu": [True],
              "dropout_rate": [[0.5, 0.5, 0.5, 0.5],
                               [0.4, 0.4, 0.4, 0.4],
                               [0.4, 0.2, 0.1, 0.05]],
              "input_dropout": [0.1, 0.2, 0.3],
              "reg": [[1e-5, 1e-5]],
              "max_constraint": [False]}

# set up the random parameter sampling
param_list = list(ParameterSampler(param_dist, n_iter=nb_search))
param_dict = [dict((k, v) for (k, v) in d.items()) for d in param_list]

with open(simdir + '/params.txt', 'w') as f:
    for i in range(len(param_dict)):
        f.write(str(param_dict[i]))
        f.write('\n')

# start the random parameter sampling
for i in range(nb_search):
    print('%d-th iteration...' % i)
    
    f = open('%s/params-%d.txt' % (simdir, i), 'w')
    f.write(str(param_dict[i]))
    f.close()
    print(param_dict[i])

    print(nb_classes, 'classes')
    print(dims, 'dims')
    print("Building model...")
    model = build_keras_model(**param_dict[i])
    model.fit(X, y, nb_epoch=nb_epoch,
              batch_size=batch_size,
              validation_split=0.15)

    log_train = model.log_train
    log_valid = model.log_validation

    print('minimum validation ll: %f at %d'
          % (np.min(log_valid), np.argmin(log_valid)))

    np.savetxt('%s/ll-%d.txt' % (simdir, i),
               np.array([model.log_train, model.log_validation]).T)
