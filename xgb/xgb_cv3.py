import subprocess
from time import sleep
import time

# "eval_metric": "mlogloss", "early_stopping_rounds": 10, "colsample_bytree": "0.6",
# "num_class": 9, "silent": 1, "nthread": 16, "min_child_weight": "4", "eta": "0.025",
# "objective", "multi:softprob", "max_depth": "12", "gamma": "0.5"}

# {"eval_metric": "mlogloss", "early_stopping_rounds": 10, "colsample_bytree": "0.5",
#  "num_class": 9, "silent": 1, "nthread": 16, "min_child_weight": "4", "eta": "0.025",
#  "objective": "multi:softprob", "max_depth": "14", "gamma": "0.5"}

etas = [0.025, 0.0125]
gammas = [0.5, 0.025]
max_depths = [12, 14]
min_child_weights = [4, 8]
colsample_bytrees = [0.7, 0.6, 0.5]
subsamples = [0.8, 0.6]

simdir = 'cv3'
num_rounds = 2000

i = 0

for ss in subsamples:
    for eta in etas:
	for gamma in gammas:
	    for md in max_depths:
		for mcw in min_child_weights:
		    for cb in colsample_bytrees:
			subprocess.call(
			    ['qsub', 'sim3',
			     simdir, str(i), str(num_rounds),
			     str(eta), str(gamma), str(md), str(mcw),
			     str(cb), str(ss)]
			)
			i += 1

    sleep(60*30)

###############################################################################
# Monitor the progress of xgb and terminate early if no logloss occurs.


def monitor_and_qdel():
    return_code = subprocess.Popen(['qstat'],
				   stderr=subprocess.STDOUT,
				   stdout=subprocess.PIPE).communicate()[0]

    qs = []

    for ln in return_code.split('\n'):
	if 'xgb3' in ln:
	    try:
		qs.append( ln.split(' ')[2])
	    except:
		pass

    qs = qs[1:]
    eps = 0.5e-6

    for q in qs:
	with open('xgb3.e%s' %q, 'rb') as f:

	    try:
		lines = f.readlines()
		l25 = float(lines[-26].split(':')[1].split('+')[0])
		l1  = float(lines[-1].split(':')[1].split('+')[0])
	    except:
		continue

	    if (l25 - l1) < eps:
		print 'xgb3.o%s' %q
		rc = subprocess.Popen(['qdel', q], stderr=subprocess.STDOUT,
				      stdout=subprocess.PIPE).communicate()[0]
		print rc


while True:
    monitor_and_qdel()
    time.sleep(60)
    

                    
