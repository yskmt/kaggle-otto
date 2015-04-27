import os
import errno
import subprocess
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn import cross_validation


call_rgf = ['perl', './call_exe.pl', './rgf1.2/bin/rgf']


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def logloss_mc(y_prob, y_true, epsilon=1e-15):
    """Multiclass logloss:
    https://github.com/ottogroup/kaggle/blob/master/benchmark.py

    Precit the probability of some model and calculate the logloss
    error for cross-validation.

    """
    # normalize
    # y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def gen_datafiles(train_csv, simdir='.', n_folds=1):
    """
    Load the train file and convert it to RGF-usable files.
    Split into n_folds files for cross validation
    """

    # import data
    train = pd.read_csv(train_csv)

    # drop ids and get labels
    labels = train.target.values
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)

    # change to numpy array
    train = train.values

    # encode labels
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)

    # shuffle first
    n, p = train.shape
    random_select = range(n)
    np.random.seed(1234)
    np.random.shuffle(random_select)
    train = train[random_select, :]
    labels = labels[random_select]

    # create cv number of files for cross validation
    kf = cross_validation.KFold(n, n_folds=n_folds,
                                shuffle=True,
                                random_state=1234)

    # save all the K-fold split data
    fn = 0
    for train_index, test_index in kf:
        X_train, X_test = train[train_index, :], train[test_index, :]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # save the training and target data for rgf
        np.savetxt(simdir + '/data/X_train-%d' % fn, X_train, fmt='%d')
        np.savetxt(simdir + '/data/X_test-%d' % fn, X_test, fmt='%d')
        np.savetxt(simdir + '/data/y_train-%d' % fn, y_train, fmt='%d')
        np.savetxt(simdir + '/data/y_test-%d' % fn, y_test, fmt='%d')
        # convert the labels to +-1
        for i in range(9):
            np.savetxt(simdir + '/data/y_train-%d-%d' % (fn, i),
                       [1 if l == i else -1 for l in y_train], fmt='%d')
            np.savetxt(simdir + '/data/y_test-%d-%d' % (fn, i),
                       [1 if l == i else -1 for l in y_test], fmt='%d')
        fn += 1


def gen_l2l3files(train_csv, simdir='.', n_folds=1):
    """
    Load the train file and convert it to RGF-usable files.
    Generate label2-label3 classifying file.
    """

    # import data
    train = pd.read_csv(train_csv)

    # drop ids and get labels
    labels = train.target.values
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)

    # change to numpy array
    train = train.values

    # encode labels
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)
    
    # extract label 2 and 3 (1 and 2 in numpy array)
    l1l2_indices = np.vstack(
        (np.argwhere(labels==1), np.argwhere(labels==2))).flatten()
    train = train[l1l2_indices, :]
    labels = labels[l1l2_indices]
    # convert the labels to +-1
    # label2(1): +1, label3(2): -1
    labels[labels==2] = -1
    
    # shuffle first
    n, p = train.shape
    random_select = range(n)
    np.random.seed(1234)
    np.random.shuffle(random_select)
    train = train[random_select, :]
    labels = labels[random_select]
    
    # create cv number of files for cross validation
    kf = cross_validation.KFold(n, n_folds=n_folds,
                                shuffle=True,
                                random_state=1234)

    # save all the K-fold split data
    fn = 0
    for train_index, test_index in kf:
        X_train, X_test = train[train_index, :], train[test_index, :]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # save the training and target data for rgf
        np.savetxt(simdir + '/data/X_train-%d' % fn, X_train, fmt='%d')
        np.savetxt(simdir + '/data/X_test-%d' % fn, X_test, fmt='%d')
        np.savetxt(simdir + '/data/y_train-%d' % fn, y_train, fmt='%d')
        np.savetxt(simdir + '/data/y_test-%d' % fn, y_test, fmt='%d')

        np.savetxt(simdir + '/data/y_train_%d-%d' % (0, fn), y_train, fmt='%d')
        np.savetxt(simdir + '/data/y_test_%d-%d' % (0, fn), y_test, fmt='%d')
        fn += 1


        
def rgf_fit(train_params, train_inp):
    """Fit the RGF model to each class separately"""

    # create train input file
    with open(train_inp, 'w') as tf:
        for k in train_params.keys():
            tf.write('%s=%s\n' % (k, str(train_params[k])))

    # RGF model fit
    return_code = subprocess.call(call_rgf + ['train', train_inp[:-4]])
    return return_code


def rgf_predict(predict_params, predict_inp):
    """Predict each class separately using RGF model"""

    # predict
    with open(predict_inp, 'w') as pf:
        for k in predict_params.keys():
            pf.write('%s=%s\n' % (k, str(predict_params[k])))

    # RGF model predict
    return_code = subprocess.call(call_rgf + ['predict', predict_inp[:-4]])

    # accuracy calculation against the training data
    # label_pred = np.loadtxt('output/otto_%d.pred' % cn)
    # target_rgf = np.loadtxt('data/target_rgf_%d' % cn)
    # acc = float(sum((label_pred * target_rgf) > 0)) / len(target_rgf)
    # print "acc: ", acc

    return return_code


def construct_ens(n_features, ypredfile, calc_acc=False, simdir='.'):
    """Construct the ensemble multi-class classifier and calcualte the
    logloss.

    Strategy: take the label with the highest score.

    """

    print "constructing the ensemble classifier..."
    label_preds = []
    for cn in range(n_features):
        label_preds.append(
            np.loadtxt(ypredfile % cn))

    label_preds = np.array(label_preds).T
    n, p = label_preds.shape

    if calc_acc:
        label_targets = []
        for cn in range(n_features):
            label_targets.append(
                np.loadtxt(simdir + '/data/target_rgf_%d' % cn))

        label_targets = np.array(label_targets).T

        label_preds_ = np.zeros((n, p))
        np.copyto(label_preds_, label_preds)

        label_preds_[label_preds < 0] = -1
        label_preds_[label_preds > 0] = +1

        for cn in range(9):
            print metrics.confusion_matrix(label_targets[:, cn], label_preds_[:, cn])

        acc = np.sum((label_preds * label_targets) > 0, axis=0) / float(n)

    label_ens = np.zeros(n)
    for i in range(n):
        label_ens[i] = np.argmax(label_preds[i, :])

    if not calc_acc:
        acc = 0

    return label_ens, acc


def calc_ll(label_ens, label_true, eps=1e-15):
    """Calculat the logloss from predicted labels.

    Note: label_ens is the array of predictions, not probabilities
    (or, the probability of these preditions are strictly 1).

    """

    n = len(label_true)
    ll = 0

    for i in range(n):
        if label_true[i] != label_ens[i]:
            ll += np.log(eps)

    return ll / (-n)


def calc_proba(n_features, ypredfile, simdir='.'):
    """
    Calculate probability matrix from the scores.
    exp(yi)/sum(exp(y)) is the transformation of individual score to proability.
    """

    label_preds = []
    for cn in range(n_features):
        label_preds.append(np.loadtxt(ypredfile %cn))

    label_preds = np.array(label_preds).T
    n, p = label_preds.shape

    proba = np.exp(label_preds) / np.sum(np.exp(label_preds), axis=1)[:, None]
    
    return proba


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
