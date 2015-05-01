import os
import errno
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def logloss_mc(model, x, y_true, epsilon=1e-15):
    """Multiclass logloss:
    https://github.com/ottogroup/kaggle/blob/master/benchmark.py

    Precit the probability of some model and calculate the logloss
    error for cross-validation.

    """

    # predict probability
    # print 'preciting probability for cv #: %d' % cv_count
    y_prob = model.predict_proba(x)

    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))

    print "Logloss: ", ll

    return ll


def calc_ll_from_proba(label_proba, label_true, eps=1e-15):
    """
    Calculate the logloss from the probability matrix.
    """

    # create a probability matrix for test labels (1s and 0s)
    N, m = label_proba.shape
    test_proba = np.zeros((N, m))
    idx = np.array(list(enumerate(label_true)), dtype=int)
    test_proba[idx[:, 0], idx[:, 1]] = 1

    logloss = -np.sum(
        test_proba * np.log(
            np.maximum(np.minimum(label_proba, 1 - 1e-15), 1e-15))) / N

    return logloss



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


def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids


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

    return clf
