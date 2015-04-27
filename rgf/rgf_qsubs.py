import subprocess


algo = 'RGF'
loss = 'Expo'
L2s = [1.0, 0.1, 0.05, 0.025, 0.01]
sL2s = [0.1, 0.01, 0.001]
mlf = 12000
ti = 1000

i = 0
for L2 in L2s:
    for sL2 in sL2s:
        simdir = 'cv_%d' %i
        subprocess.call(['qsub', 'sim', simdir, algo, loss, str(L2), str(sL2),
                         str(mlf), str(ti)])
        i += 1
