import numpy as np
import matplotlib.pyplot as plt

eval_ll = []
train_ll = []
eval_std = []
train_std = []

fname = 'log/xgcv_01'

with open(fname) as f:
    for line in f:
        try:
            l = line.split('\t')
            le = l[1].split(':')[-1].split('+')
            lt = l[2].split(':')[-1].split('+')
            eval_ll.append(le[0])
            eval_std.append(le[1])
            
            train_ll.append(lt[0])
            train_std.append(lt[1])
        except:
            print l
        

eval_ll = np.array(eval_ll, dtype=float)
train_ll = np.array(train_ll, dtype=float)




plt.plot(eval_ll, label='eval-mlogloss')
plt.plot(train_ll, label='train-mlogloss')
plt.legend(loc='best')
plt.show()
