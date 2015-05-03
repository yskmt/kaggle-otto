import subprocess


def monitor_and_qdel():
    return_code = subprocess.Popen(['qstat'],
				   stderr=subprocess.STDOUT,
				   stdout=subprocess.PIPE).communicate()[0]

    qs = []

    for ln in return_code.split('\n'):
	try:
	    qs.append( ln.split(' ')[2])
	except:
	    pass

    qs = qs[1:]
    eps = 0.5e-6

    for q in qs:
	with open('xgb2.o%s' %q, 'rb') as f:

	    try:
		lines = f.readlines()
		l25 = float(lines[-26].split(':')[1].split('+')[0])
		l1  = float(lines[-1].split(':')[1].split('+')[0])
	    except:
		continue

	    if (l25 - l1) < eps:
		print 'xgb2.o%s' %q
		rc = subprocess.Popen(['qdel', q], stderr=subprocess.STDOUT,
				      stdout=subprocess.PIPE).communicate()[0]
		print rc

import time

while True:
    monitor_and_qdel()
    time.sleep(60)
    
