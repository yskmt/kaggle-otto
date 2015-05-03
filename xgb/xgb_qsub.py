import subprocess
from time import sleep

etas = [0.1, 0.05, 0.025]
gammas = [1, 0.5, 0.25]
max_depths = [8, 10, 12]
min_child_weights = [1, 2, 4]
colsample_bytrees = [1, 0.8, 0.6]

simdir = 'cv'
num_rounds = 2000

i = 0

for eta in etas:
    for gamma in gammas:
        for md in max_depths:
            for mcw in min_child_weights:
                for cb in colsample_bytrees:
                    subprocess.call(
                        ['qsub', 'sim',
                         simdir, str(i), str(num_rounds),
                         str(eta), str(gamma), str(md), str(mcw), str(cb)]
                    )
                    i += 1

        sleep(60*15)

                    
