import sys
from cgcnniap.iap import CGCNN_IAP
from ase.build import bulk, make_supercell
from ase.io import read,write
import numpy as np
import time

modelpath = sys.argv[1]
datapath = sys.argv[2]

iap = CGCNN_IAP(modelpath,datapath)

# note this oddly doesn't work for a 1 atom crystal structure
dummy = bulk('Cu', 'fcc', a=3.6)
dummy = make_supercell(dummy, 10*np.eye(3))
print(len(dummy))
#dummy = read('../sample-regression/1000041.cif')

energy = iap.predict(dummy)

start =time.time()
for i in range(10000):
    dummy.rattle(stdev=0.2)
    #print(energy[0][0])
end = time.time()
print(end-start)
