"""
Try Ensemble of NN with different parameters...

"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from keras_utils import (load_data, keras_bagging,
                         preprocess_data, preprocess_labels)

np.random.seed(1234)  # for reproducibility

# simulation parameters
simname = 'bagg'
batch_size = 256
nb_epoch = 10
n_folds = 4
lays = [100] * 4  # layer_sizes

print("Loading data...")
X, labels = load_data('../data/train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels, categorical=False)

dims = X.shape[1]
nb_classes = max(y) + 1

params = {"nb_classes": 9, "dims": dims,
          "layer_size": lays,
          "opt": "adagrad", "sgd_lr": 0.1, "sgd_decay": 0.1,
          "sgd_mom": 0.9, "sgd_nesterov": False,
          "activation_func": "relu",
          "weight_ini": "glorot_uniform",
          "batchnorm": False, "prelu": False,
          "dropout_rate": [0.5, 0.5, 0.5, 0.5],
          "input_dropout": 0.2,
          "reg": [1e-5, 1e-5],
          "max_constraint": False}

# K-fold cross validation on the Baggin NN model
probas, lls = keras_bagging(simname, 0, params, X, y, n_folds,
                            nb_epoch, batch_size)
