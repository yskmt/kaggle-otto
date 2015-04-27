"""
RandomForestClassifier
sklearn: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto
"""

import sys
import subprocess
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing


if len(sys.argv) < 3:
    print "number of arguments: 2"
    sys.exit(1)

cn = int(sys.argv[1])
mn = int(sys.argv[2])


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


def gen_datafiles(train_csv):
    """
    Load the train file and convert it to RGF-usable files.
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

    # shuffle
    n, p = train.shape
    random_select = range(n)
    np.random.shuffle(random_select)
    train = train[random_select, :]
    labels = labels[random_select]

    # save the training and target data for rgf
    np.savetxt('data/train_rgf', train)
    np.savetxt('data/label_rgf', labels)
    for i in range(9):
        np.savetxt('data/target_rgf_%d' % (i + 1),
                   [1 if l == i else -1 for l in labels], fmt='%d')


def rgf_fit(cn, test_interval, max_leaf_forest):
    # fit the RGF model to each class separately
    # create train input file
    train_params = {'reg_L2': 1e-20,
                    'algorithm': 'RGF',
                    'loss': 'Log',  # LS|Expo|Log
                    'test_interval': test_interval,
                    'max_leaf_forest': max_leaf_forest,
                    'train_x_fn': 'data/train_rgf',
                    'train_y_fn': 'data/target_rgf_%d' % cn,
                    'model_fn_prefix': 'output/otto_%d.model' % cn}
    train_inp = 'input/train_%d.inp' % cn
    with open(train_inp, 'w') as tf:
        for k in train_params.keys():
            tf.write('%s=%s\n' % (k, str(train_params[k])))

    # RGF model fit
    # os.chdir('/Users/ysakamoto/Projects/rgf')
    return_code = subprocess.call(['perl', './call_exe.pl',
                                   './rgf1.2/bin/rgf', 'train',
                                   train_inp[:-4]])


def rgf_predict(cn, mn):
    """
    cn: feature (label) number
    mn: model number
    """

    # predict
    # write predict input file
    predict_params = {'test_x_fn': 'data/train_rgf',
                      'model_fn': 'output/otto_%d.model-%02d' % (cn, mn),
                      'prediction_fn': 'output/otto_%d.pred-%02d' % (cn, mn)}
    predict_inp = 'input/predict_%d.inp' % cn
    with open(predict_inp, 'w') as pf:
        for k in predict_params.keys():
            pf.write('%s=%s\n' % (k, str(predict_params[k])))

    # RGF model predict
    return_code = subprocess.call(['perl', './call_exe.pl',
                                   './rgf1.2/bin/rgf', 'predict',
                                   predict_inp[:-4]])

    # accuracy calculation against the training data
    # label_pred = np.loadtxt('output/otto_%d.pred' % cn)
    # target_rgf = np.loadtxt('data/target_rgf_%d' % cn)
    # acc = float(sum((label_pred * target_rgf) > 0)) / len(target_rgf)
    # print "acc: ", acc


def construct_ens(mn, calc_acc=False):
    """Construct the ensemble multi-class classifier and calcualte the
    logloss.

    Strategy: take the label with the highest score.

    """

    print "constructing the ensemble classifier..."
    label_preds = []
    for cn in range(1, 10):
        label_preds.append(np.loadtxt('output/otto_%d.pred-%02d' % (cn, mn)))
            
    label_preds = np.array(label_preds).T
    n, p = label_preds.shape

    if calc_acc:
        label_targets = []
        for cn in range(1, 10):
            label_targets.append(np.loadtxt('data/target_rgf_%d' %cn))

        label_targets = np.array(label_targets).T

        label_preds_ = np.zeros((n,p))
        np.copyto(label_preds_, label_preds)
        
        label_preds_[label_preds<0] = -1
        label_preds_[label_preds>0] = +1
        
        for cn in range(9):
            print metrics.confusion_matrix(label_targets[:, cn], label_preds_[:,cn])
        
        accs = np.sum((label_preds*label_targets)>0, axis=0)/float(n)

    label_ens = np.zeros(n)
    for i in range(n):
        label_ens[i] = np.argmax(label_preds[i, :])

    if not calc_acc:
        acc = 0
        
    return label_ens, acc


def calc_ll(label_ens, label_true, eps=1e-15):
    """Calculat the logloss.

    Note: label_ens is the array of predictions, not probabilities
    (or, the probability of these preditions are strictly 1).

    """

    n = len(label_true)
    ll = 0

    for i in range(n):
        if label_true[i] != label_ens[i]:
            ll += np.log(eps)

    return ll / (-n)


print "cleaning up the data..."
train_csv = 'data/train.csv'
# gen_datafiles(train_csv)

# Note: RGF currently only supports binary classificaiton. Thus, we
# run the predictive simulations as many times as the number of
# categories (9).

ti = 1000
mlf = 10000
# mn = int(mlf / ti)

# # fit
# for cn in range(1, 10):
#     rgf_fit(cn, ti, mlf)

# predict
lls = []
for mn in range(1, 11):
    # predict each label
    for cn in range(1, 10):
        rgf_predict(cn, mn)

    # construct an ensemble model
    label_true = np.loadtxt('data/label_rgf')
    label_ens, accs = construct_ens(mn, False)
    print accs
    ll = calc_ll(label_ens, label_true)
    lls.append(ll)
    
    print 'mlogloss: ', ll

    # confusion matrix plot:
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    print 'confusion matrix: '
    print metrics.confusion_matrix(label_true, label_ens)





# mlogloss:  7.0464060443
# confusion matrix: 
# [[  864   123    19     1     6   175    74   283   384]
#  [   13 13895  1787   176    25    32   111    52    31]
#  [    2  3969  3689   134     0    14   138    42    16]
#  [    1  1070   414  1060    21    82    29    10     4]
#  [    5    65     6     2  2646     7     4     2     2]
#  [   71   150    32    23     3 13313   166   207   170]
#  [   64   338   211    40    14   208  1784   162    18]
#  [  128   132    39     3     5   189    89  7734   145]
#  [  131   152    15     3     6   161    39   179  4269]]

# mlogloss:  8.94699322593
# confusion matrix: 
# [[  341   140    10     0     9   246    51   600   532]
#  [    4 14536  1174    26   165    69    69    59    20]
#  [    1  5583  2051    30   104    29   128    63    15]
#  [    0  1797   281   350    22   183    49     7     2]
#  [    1   138     4     1  2582     5     0     5     3]
#  [   34   211    24    11     8 13159   154   329   205]
#  [   28   525   214    13    15   290  1355   353    46]
#  [   55   165    41     0    10   399    42  7614   138]
#  [   68   266     9     0    20   266    17   448  3861]]
