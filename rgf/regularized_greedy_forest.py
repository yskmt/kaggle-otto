"""RGF (Regularized Greedy Forst) Classifier
http://stat.rutgers.edu/home/tzhang/software/rgf/
Otto Group product classification challenge @ Kaggle

__author__ : Yusuke Sakamoto

Run the RGF model fit and predict for all the cross-validation folds
for each label number and model number specified.

Note: RGF currently only supports binary classificaiton. Thus, we run
the predictive simulations as many times as the number of labels
(9). Thus, these binary classification (training, prediction) can be
paralleliszed.

"""

import subprocess
import os
import errno

import numpy as np
from joblib import Parallel, delayed


call_rgf = ['perl', './call_exe.pl', './rgf1.2/bin/rgf']


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def _fit_each_label(train_params, train_inp,
                    y_file, model_file, lbn):
    train_params['train_y_fn'] = y_file + '-' + str(lbn)
    train_params['model_fn_prefix'] = model_file + '-' + str(lbn)
    train_inp = train_inp + '-' + str(lbn) + '.inp'

    # create train input file
    with open(train_inp, 'w') as tf:
        for k in train_params.keys():
            if train_params[k] is not None:
                tf.write('%s=%s\n' % (k, str(train_params[k])))

    # RGF model fit
    subprocess.call(call_rgf + ['train', train_inp[:-4]])

    return


def _predict_each_label(X_file, model_file, predict_file, lbn,
                        model_number, predict_inp):

    predict_params = {'test_x_fn': X_file,
                      'model_fn':
                      model_file + '-%d-%02d' % (lbn, model_number),
                      'prediction_fn':
                      predict_file + '-%d-%02d' % (lbn, model_number)}

    _predict_inp = predict_inp + '-%d-%02d' % (lbn, model_number)\
        + '.inp'

    # write predict input file
    with open(_predict_inp, 'w') as pf:
        for k in predict_params.keys():
            pf.write('%s=%s\n' % (k, str(predict_params[k])))

    # RGF predict!
    subprocess.call(call_rgf + ['predict', _predict_inp[:-4]])


class RegularizedGreedyForestClassifier():

    def __init__(self, simdir='.', loss='LS', algorithm='RGF',
                 max_leaf_forest=10000, test_interval=1000,
                 reg_L2=1.0, reg_sL2=None,
                 n_labels=2, n_jobs=1):
        self.train_params = {
            'loss': loss,
            'algorithm': algorithm,
            'max_leaf_forest': max_leaf_forest,
            'test_interval': test_interval,
            'reg_sL2': reg_sL2,
            'reg_L2': reg_L2,
        }

        self.simdir = simdir
        self.train_inp = self.simdir + '/input/train'
        self.predict_inp = self.simdir + '/input/predict'
        self.n_labels = n_labels
        self.proba = None
        self.n_models = int(max_leaf_forest / test_interval)
        self.n_jobs = n_jobs
        
    def fit(self, X_file, y_file, model_file):
        """Fit the RGF model to each class separately"""

        mkdir_p(self.simdir)
        mkdir_p(self.simdir + '/input')
        mkdir_p(self.simdir + '/output')
        mkdir_p(self.simdir + '/model')

        self.train_params['train_x_fn'] = X_file
        # for i in range(self.n_labels):
        # self.fit_each_label(y_file, model_file, i)

        Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_each_label)(
                self.train_params, self.train_inp, y_file, model_file, i)
            for i in range(self.n_labels))

        return self

    # def fit_each_label(self, y_file, model_file, lbn):
    #     self.train_params['train_y_fn'] = y_file + '-' + str(lbn)
    #     self.train_params['model_fn_prefix'] = model_file + '-' + str(lbn)
    #     train_inp = self.train_inp + '-' + str(lbn) + '.inp'

    #     # create train input file
    #     with open(train_inp, 'w') as tf:
    #         for k in self.train_params.keys():
    #             if self.train_params[k] is not None:
    #                 tf.write('%s=%s\n' % (k, str(self.train_params[k])))

    #     # RGF model fit
    #     subprocess.call(call_rgf + ['train', train_inp[:-4]])

    #     return self

    def predict(self, X_file, model_file, predict_file,
                model_number=None):
        if self.proba is None:
            self.predict_proba(self, X_file, model_file, predict_file,
                               model_number)

        n, p = self.proba.shape
        label_ens = np.zeros(n)
        for i in range(n):
            label_ens[i] = np.argmax(self.proba[i, :])

        return label_ens

    def predict_proba(self, X_file, model_file, predict_file,
                      model_number=None):

        if model_number is None:
            model_number = self.n_models

        Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_each_label)(X_file, model_file, predict_file,
                                         i, model_number, self.predict_inp)
            for i in range(self.n_labels))

        # for i in range(self.n_labels):
        #     self.predict_each_label(X_file, model_file, predict_file,
        #                             i, model_number=model_number)

        # read output file
        self.proba = self.calc_proba(predict_file, model_number=model_number)

        return self.proba

    def predict_each_label(self, X_file, model_file, predict_file,
                           lbn, model_number=None):

        if model_number is None:
            model_number = self.n_models
        self.predict_params = {'test_x_fn': X_file,
                               'model_fn':
                               model_file + '-%d-%02d' % (lbn, model_number),
                               'prediction_fn':
                               predict_file + '-%d-%02d' % (lbn, model_number)}

        predict_inp = self.predict_inp + '-%d-%02d' % (lbn, model_number)\
            + '.inp'

        # write predict input file
        with open(predict_inp, 'w') as pf:
            for k in self.predict_params.keys():
                pf.write('%s=%s\n' % (k, str(self.predict_params[k])))

        # RGF predict!
        subprocess.call(call_rgf + ['predict', predict_inp[:-4]])

    def calc_proba(self, predict_file, model_number=None):
        """
        Calculate probability matrix from the scores.
        exp(yi)/sum(exp(y)) is the transformation of individual score to proability.
        """

        if model_number is None:
            model_number = self.n_models

        label_preds = []
        for lbn in range(self.n_labels):
            label_preds.append(
                np.loadtxt(predict_file + '-%d-%02d' % (lbn, model_number)))

        label_preds = np.array(label_preds).T
        n, p = label_preds.shape

        proba = np.exp(label_preds) / \
            np.sum(np.exp(label_preds), axis=1)[:, None]

        return proba
