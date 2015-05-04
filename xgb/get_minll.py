import numpy as np

from os import listdir
from os.path import isfile, join


files = [ f for f in listdir('.') if isfile(join('.',f)) ]
files.sort()

# files = [f for f in files if f.startswith('xgb.o') or f.startswith('xgb2.o')]
# files = [f for f in files if f.startswith('xgb2.o')]
files = [f for f in files if f.startswith('xgb3.e')]

print files

lls = []
for f in files:
    with open(f, 'r') as fx:
	lines = fx.readlines()
	try:
	    if lines[-1].startswith('DONE'):
		lls.append(float(lines[-5].split(':')[1].split('+')[0]))
	    else:
		lls.append(float(lines[-1].split(':')[1].split('+')[0]))
	except:
	    lls.append(float(lines[-7].split(':')[1].split('+')[0]))

lls = np.array(lls)

print ''
print lls.argmin(), lls.min()
print files[lls.argmin()]
