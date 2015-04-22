"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle

__author__ : Abhishek Thakur
"""

import sys
if '/Users/ysakamoto/Projects/scikit-learn' not in sys.path:
    sys.path.append('/Users/ysakamoto/Projects/scikit-learn')

import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

'''NOTE:

1. Loss function for GradientBoostingClassifier: why don't I use log
loss, as it is the measure of the success in this competition.
2. More robust cross-validation (10-fold).
3. Maybe buy Amazon instance.
    - [Star Cluster](http://star.mit.edu/cluster/)
'''

# calculate the logloss score (smaller better)
def calculate_logloss(labels_proba, labels_test):

    N, m = labels_proba.shape
    test_proba = np.zeros((N, m))
    idx = np.array(list(enumerate(labels_test)))
    test_proba[idx[:, 0], idx[:, 1]] = 1

    logloss = -np.sum(
        test_proba * np.log(
            np.maximum(np.minimum(labels_proba, 1 - 1e-15), 1e-15))) / N

    return logloss


# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
# test = test.drop('id', axis=1)

n, p = train.shape
N = n / 2

# splitting the data by random select
random_select = range(n)
np.random.shuffle(random_select)

train_ = train.iloc[random_select[:N]]
test_ = train.loc[random_select[N:]]

train = train_
test = test_
labels_train = labels[random_select[:N]]
labels_test = labels[random_select[N:]]

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels
lbl_enc = preprocessing.LabelEncoder()
labels_train = lbl_enc.fit_transform(labels_train)
labels_test = lbl_enc.fit_transform(labels_test)

# train a random forest classifier
# clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
# clf = svm.SVC(gamma=0.001, C=100., probability=True)
# clf = RandomForestClassifier(n_estimators=200, max_depth=None,
# min_samples_split=1)

learning_rate = [0.05, 0.025]
subsample = [0.8, 0.6]
lls = np.zeros((len(learning_rate), len(subsample)))
for i in range(len(learning_rate)):
    for j in range(len(subsample)):
        clf = GradientBoostingClassifier(learning_rate=learning_rate[i],
                                         n_estimators=2000,
                                         verbose=1,
                                         subsample=subsample[j])
        clf.fit(train, labels_train)

        # predict on test set
        labels_proba = clf.predict_proba(test)
        # labels_pred = clf.predict(test)
        lls[i, j] = calculate_logloss(labels_proba, labels_test)

print lls


# learning_rate = [0.2, 0.1, 0.05]
# subsample = [1.0, 0.5, 0.25]
# max_features = auto
# n_estimators=200
# [[ 0.59024639  0.67890676  5.99474064]
#  [ 0.58396747  0.59919191  0.72376028]
#  [ 0.63113617  0.63160567  0.63892663]]

# n_estimators=800
# [[  0.61950479  26.616035    30.50093567]
#  [  0.55407165   1.94580965   7.73248527]
#  [  0.54707369   0.56781943   0.90090593]]

# n_estimators=1600
# [[  0.68293461  22.44646488  27.58502746]
#  [  0.57150196   9.09286839  31.84278929]
#  [  0.54250501   0.67395222   3.46564126]]

# learning_rate = [0.4, 0.2, 0.1, 0.05]
# subsample = [1.0, 0.8, 0.6]
# n_estimators = 1000
# max_features = None
# [[ 1.11440548  2.04432256  1.07085385]
#  [ 0.68416995  1.2353997   0.96037465]
#  [ 0.60521535  0.61459463  0.69588081]
#  [ 0.57314594  0.56486122  0.57236782]]

# [[ 0.919027    1.56585946  1.63269389]
#  [ 0.68280459  0.7583199   1.36552339]
#  [ 0.59368556  0.62216994  0.80985971]
#  [ 0.55851679  0.55929985  0.5795528 ]]


# hajnal
# learning_rate = [0.05, 0.025]
# subsample = [0.8, 0.6]
# n_estimators=2000
# [[ 0.56804766  0.62039549]
#  [ 0.54620701  0.55494456]]

# euclid
# [[0.573379789132, 0.564612563905],
#  [0.545428083892, 0.550226135432]]

# ###############################################################################
# # PLS regression
# N, m = train.shape
# train_proba = np.zeros((N, m))
# idx = np.array(list(enumerate(labels_train)))
# train_proba[idx[:, 0], idx[:, 1]] = 1

# for i in range(9):
#     pls2 = PLSRegression(n_components=5, algorithm="kernel")
#     pls2.fit(train, train_proba[:, i])
#     labels_proba[:, i] = pls2.predict(test).ravel()
# ###############################################################################


# # plot predicted target against test target
# plt.plot(labels_test, labels_pred, '.')
# plt.show()


# create submission file
# preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
# preds.to_csv('benchmark.csv', index_label='id')


# print "logloss score: ", logloss
