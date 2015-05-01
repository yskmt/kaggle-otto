
# coding: utf-8

# # Otto Group Product Classification Challenge using nolearn/lasagne

# This short notebook is meant to help you getting started with nolearn and lasagne in order to train a neural net and make a submission to the Otto Group Product Classification Challenge.
#
# * [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)
# * [Get the notebook from the Otto Group repository](https://github.com/ottogroup)
# * [Nolearn repository](https://github.com/dnouri/nolearn)
# * [Lasagne repository](https://github.com/benanne/Lasagne)
# * [A nolearn/lasagne tutorial for convolutional nets](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)

# Imports
import os
import sys

from sklearn import cross_validation
from sklearn import metrics

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

pardir = os.path.realpath('..')
if pardir not in sys.path:
    sys.path.append(pardir)


from otto_utils import (load_train_data, load_test_data,
                        mkdir_p, calc_ll_from_proba)


X, y, encoder, scaler = load_train_data('../data/train.csv')
X_test, ids = load_test_data('../data/test.csv', scaler)

num_classes = len(encoder.classes_)
num_features = X.shape[1]
n = X.shape[0]

n_folds = 5

# create cv number of files for cross validation
kf = cross_validation.KFold(n, n_folds=n_folds,
                            shuffle=True,
                            random_state=1234)


# Train Neural Net
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 dense0_num_units=200,
                 dropout_p=0.5,
                 dense1_num_units=200,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=20)

ll = []
ncv = 0
for train_index, test_index in kf:
    print "cross-validation: %dth fold..." % ncv

    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]

    # mkdir_p(cv_params['simdir'] + '/data')
    # np.savetxt(cv_params['simdir'] + '/data/y_test-%d' % ncv, y_test)

    print "fitting nn model.."
    net0.fit(X_train, y_train)

    print "predicting probabilities using nn model..."
    proba = net0.predict_proba(X_test)
    ll.append(calc_ll_from_proba(proba, y_test))

    print metrics.confusion_matrix(
        y_test.astype(int), np.argmax(proba, axis=1).astype(int))

    # print "saving svc model..."
    # joblib.dump(net0, cv_params['simdir'] + '/svc_%d.pkl' % ncv)

    ncv += 1

ll = np.array(ll)
print "logloss: ", ll
    
# make_submission(net0, X_test, ids, encoder)
