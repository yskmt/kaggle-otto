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
from sklearn.ensemble import ExtraTreesClassifier


def load_data(filename):

    print "cleaning the train data..."
    filename_ = filename.replace('.csv', '_.csv')    
    with open(filename) as f:
        with open(filename_, 'w') as g:
            for line in f:
                g.write(line.replace('Class_', ''))
    
    print "loading data from file %s..." %filename_
    r = np.loadtxt(filename_, delimiter=',', skiprows=1)

    return filename_, r

    
train_file = 'train.csv'
train_file_, r = load_data(train_file)

# test_file = 'test.csv'
# test_file_ = 'test_.csv'

# id: starting from 0
ids = np.asarray(r[:, 0], dtype=int) - 1
data = r[:, 1:-1]
target = r[:, -1]

n, p = data.shape
m = 9  # number of labels

# randomly select ids to split training and test sets
random_select = range(n)
np.random.shuffle(random_select)
# random_select = np.random.choice(range(0,n), n, replace=False)
ids_train = random_select[:n / 2]
ids_test = random_select[n / 2:]

# split train and test
train_data = data[ids_train, :]
test_data = data[ids_test, :]

train_target = target[ids_train]
test_target = target[ids_test]
N = len(test_target)

# fit model and predict!
# clf = svm.SVC(gamma=0.001, C=100.)
clf = ExtraTreesClassifier(n_estimators=1000, max_depth=None,
                           min_samples_split=1, random_state=None)
time0 = time.time()
print "fitting train data and target..."
clf.fit(train_data, train_target)
print "predicting the test target"
pred_target = clf.predict(test_data)
print "time elasped for prediction: ", time.time() - time0

# plot predicted target against test target
plt.plot(test_target, pred_target, '.')
plt.show()

# now create a probability matrix
target_prob = np.zeros((N, m))
test_prob = np.zeros((N, m))
for i in range(N):
    target_prob[i, int(pred_target[i]) - 1] = 1.0
    test_prob[i, int(test_target[i]) - 1] = 1.0

# calculate the logloss score (smaller better)
logloss = 0.0
for i in range(N):
    for j in range(m):
        yij = test_prob[i, j]
        logloss -= yij * np.log(max(min(target_prob[i, j], 1 - 1e-15), 1e-15))

logloss /= N

print "logloss score: ", logloss


print "writing the results to file..."
submission_file = 'submission.csv'
results = np.zeros((N,m+1), dtype=int)
results[:,0] = np.array(range(1, N+1))
results[:,1:] = target_prob

header = 'id,Class_1,Class_2,Class_3,Class_4,'\
         'Class_5,Class_6,Class_7,Class_8,Class_9'
np.savetxt(submission_file, results, delimiter=',', header=header, fmt='%d',
           comments='')
