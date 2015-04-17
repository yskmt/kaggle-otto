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
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt


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
clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             min_samples_split=1)
clf.fit(train, labels_train)

# predict on test set
labels_proba = clf.predict_proba(test)
labels_pred = clf.predict(test)



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


# plot predicted target against test target
# plt.plot(labels_test, labels_pred, '.')
# plt.show()


# create submission file
# preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
# preds.to_csv('benchmark.csv', index_label='id')


# calculate the logloss score (smaller better)
N, m = labels_proba.shape
test_proba = np.zeros((N, m))
idx = np.array(list(enumerate(labels_test)))
test_proba[idx[:, 0], idx[:, 1]] = 1

logloss = -np.sum(
    test_proba * np.log(
        np.maximum(np.minimum(labels_proba, 1 - 1e-15), 1e-15))) / N

# logloss = 0.0
# for i in range(N):
#     for j in range(m):
#         yij = test_proba[i, j]
#         logloss -= yij * np.log(max(min(labels_proba[i, j], 1 - 1e-15), 1e-15))

# logloss /= N

print "logloss score: ", logloss
