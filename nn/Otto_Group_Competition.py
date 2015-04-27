
# coding: utf-8

# # Otto Group Product Classification Challenge using nolearn/lasagne

# This short notebook is meant to help you getting started with nolearn and lasagne in order to train a neural net and make a submission to the Otto Group Product Classification Challenge.
# 
# * [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)
# * [Get the notebook from the Otto Group repository](https://github.com/ottogroup)
# * [Nolearn repository](https://github.com/dnouri/nolearn)
# * [Lasagne repository](https://github.com/benanne/Lasagne)
# * [A nolearn/lasagne tutorial for convolutional nets](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)

# ## Imports

# In[1]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[2]:

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


# ## Utility functions

# In[3]:

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler


# In[4]:

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids


# In[5]:

def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))


# ## Load Data

# In[6]:

X, y, encoder, scaler = load_train_data('data/train.csv')


# In[7]:

X_test, ids = load_test_data('data/test.csv', scaler)


# In[8]:

num_classes = len(encoder.classes_)
num_features = X.shape[1]


# ## Train Neural Net

# In[9]:

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]


# In[10]:

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


# In[11]:

net0.fit(X, y)


# ## Prepare Submission File

# In[12]:

make_submission(net0, X_test, ids, encoder)

