import subprocess


Cs = [0.01, 0.1, 1, 10]
kernels = ['rbf', 'sigmoid']
gammas = [0, 0.5, 1.0]
simdir = 'cv'

i = 0
for kernel in kernels:
    for C in Cs:
        for gamma in gammas:
            
            subprocess.call(['qsub', 'sim', simdir, str(i),
                             str(kernel), str(C), str(gamma)])
            i += 1
