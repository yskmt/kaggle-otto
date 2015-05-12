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


np.random.seed(1337) # for reproducibility
# np.random.seed(int(time()))  # for reproducibility

batch_size = 16
nb_epoch = 100

params = {'opt': 'sgd', 'dropout_rate': 0.75, 'nb_classes': 9,
          'activation_func': 'tanh', 'layer_size': 1024, 'sgd_mom':
          0.9, 'sgd_decay': 0.0, 'weight_ini': 'glorot_uniform',
          'sgd_nesterov': True, 'batchnorm': True, 'nb_layers': 1,
          'sgd_lr': 0.01, 'prelu': True , 'l2_reg': 0.0}

simdir = 'manual'

from os import listdir
from os.path import isfile, join
onlyfiles = [ f for f in listdir(simdir) if isfile(join(simdir,f)) ]

i = len(onlyfiles)

f = open('%s/params-%d.txt' % (simdir, i), 'w')
f.write(str(params))
f.close()
print(params)

ll = []
print(nb_classes, 'classes')
print(dims, 'dims')
print("Building model...")
model = build_keras_model(**params)
model.fit(X, y, nb_epoch=nb_epoch,
          batch_size=batch_size,
          validation_split=0.15)

log_train = model.log_train
log_valid = model.log_validation

print('minimum validation ll: %f at %d'
      % (np.min(log_valid), np.argmin(log_valid)))

np.savetxt('%s/ll-%d.txt' % (simdir, i),
           np.array([model.log_train, model.log_validation]).T)
