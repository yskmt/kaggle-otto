import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random

if '/Users/ysakamoto/Projects/scikit-learn' not in sys.path:
    sys.path.append('/Users/ysakamoto/Projects/scikit-learn')

# from sklearn.cross_decomposition import PLSRegression
# from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def load_data(filename):

    print "cleaning the train data..."
    filename_ = filename.replace('.csv', '_.csv')
    with open(filename) as f:
        with open(filename_, 'w') as g:
            for line in f:
                g.write(line.replace('Class_', ''))

    print "loading data from file %s..." % filename_
    r = np.loadtxt(filename_, delimiter=',', skiprows=1)

    return filename_, r

# load data
train_file = 'train.csv'
train_file_, train_data = load_data(train_file)

test_file = 'test.csv'
test_file_, test_data = load_data(test_file)

# clean up the data
train_id = np.asarray(train_data[:, 0], dtype=int) - 1
train_feat = train_data[:, 1:-1]
train_target = train_data[:, -1]

test_id = np.asarray(test_data[:, 0], dtype=int) - 1
test_feat = test_data[:, 1:]

M = 9  # number of labels
N = len(test_id)

# fit classifier model and predict!
# clf = svm.SVC(gamma=0.001, C=100.)
clf = RandomForestClassifier(n_estimators=10, max_depth=None,
                             min_samples_split=1, random_state=0)
time0 = time.time()
print "fitting train data and target..."
clf.fit(train_feat, train_target)
print "predicting the test target"
test_target_pred = clf.predict(test_feat)
print "time elasped for prediction: ", time.time() - time0

# now create a probability matrix
target_prob = np.zeros((N, M))
for i in range(N):
    target_prob[i, int(test_target_pred[i]) - 1] = 1.0

# write the results
print "writing the results to file..."
submission_file = 'submission.csv'
results = np.zeros((N, M + 1), dtype=int)
results[:, 0] = np.array(range(1, N + 1))
results[:, 1:] = target_prob

header = 'id,Class_1,Class_2,Class_3,Class_4,'\
         'Class_5,Class_6,Class_7,Class_8,Class_9'
np.savetxt(submission_file, results, delimiter=',', header=header, fmt='%d',
           comments='')
