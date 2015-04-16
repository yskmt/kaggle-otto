import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random

if '/Users/ysakamoto/Projects/scikit-learn' not in sys.path:
    sys.path.append('/Users/ysakamoto/Projects/scikit-learn')

from sklearn.cross_decomposition import PLSRegression
from sklearn import linear_model

train_file = 'train.csv'
train_file_ = 'train_.csv'

with open(train_file) as f:
    with open(train_file_p, 'w') as g:
        for line in f:
            g.write(line.replace('Class_', ''))


r = np.loadtxt(train_file_, delimiter=',', skiprows=1)

# id: starting from 0
ids = np.asarray(r[:, 0], dtype=int)-1
data = r[:, 1:]
targets = r[:, -1]

n, p = data.shape
m = 9 # number of labels

# make the targets into 9-column matrix, each represents labels
# note: label range is 0-8, different from the data
targets_labeled = np.zeros((n, m))
for i in range(n):
    targets_labeled[i, int(targets[i])-1] = 1

# randomly select ids to split training and test sets
random_select = np.random.choice(range(0,n), n, replace=False)
ids_train = random_select[:n/2]
ids_test = random_select[n/2:]

# split train and test
data_train = data[ids_train, :]
data_test = data[ids_test, :]

targets_train = targets_labeled[ids_train, :]
targets_test = targets_labeled[ids_test, :]
N = len(targets_test)

# target probablity matrix
targets_prob = np.zeros((N, m))

model = PLSRegression(n_components=10, algorithm="kernel")
# model = linear_model.Ridge(alpha=.5)

# from sklearn import svm
# model = svm.SVC(gamma=0.001, C=100.)

# model.fit(data_train, targets[ids_train])
# target_pred = model.predict(data_test)

for j in range(m):
    time0 = time.time()

    model.fit(data_train, targets_train[:, j])
    targets_prob[:, j] = model.predict(data_test).ravel()
    
    print time.time()-time0

# get the label with max probability
targets_pred = np.zeros(N)
for i in range(N):
    targets_pred[i] = np.argmax(targets_prob[i, :])+1  # +1 at the end

plt.plot(data[ids_test, -1], targets_pred+1, '.')
plt.show()

logloss = 0.0
for i in range(N):
    for j in range(m):
        yij = int(int(targets_test[i, j]) == j)
        logloss -= yij * np.log(max(min(targets_prob[i,j], 1-1e-15), 1e-15))

logloss /= N

print logloss
