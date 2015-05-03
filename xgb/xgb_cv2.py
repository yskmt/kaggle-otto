import subprocess
from time import sleep

# "eval_metric": "mlogloss", "early_stopping_rounds": 10, "colsample_bytree": "0.6",
# "num_class": 9, "silent": 1, "nthread": 16, "min_child_weight": "4", "eta": "0.025",
# "objective", "multi:softprob", "max_depth": "12", "gamma": "0.5"}


etas = [0.025, 0.0125]
gammas = [0.5, 0.025]
max_depths = [12, 14]
min_child_weights = [4, 8]
colsample_bytrees = [0.7, 0.6, 0.5]
subsamples = [1.0]

simdir = 'cv2'
num_rounds = 2000

i = 0

for eta in etas:
    for gamma in gammas:
        for md in max_depths:
            for mcw in min_child_weights:
                for cb in colsample_bytrees:
                    subprocess.call(
                        ['qsub', 'sim2',
                         simdir, str(i), str(num_rounds),
                         str(eta), str(gamma), str(md), str(mcw), str(cb)]
                    )
                    i += 1

        sleep(60*30)

                    
