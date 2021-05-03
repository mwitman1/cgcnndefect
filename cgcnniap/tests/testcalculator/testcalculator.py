#! /usr/bin/env python3

import sys
import numpy as np
import time
from copy import deepcopy

import ase
from ase.build import bulk, make_supercell
from ase.io import read,write

from cgcnniap.iap import CGCNN_IAP
from cgcnniap.utils.calculator import CGCNNCalculator

modelpath = sys.argv[1]
datapath = sys.argv[2]

# note this oddly doesn't work for a 1 atom crystal structure
dummy = bulk('Mg', 'fcc', a=3.6)
dummy = make_supercell(dummy, 20*np.eye(3))

# can directly predict through iap.predict(atoms) 
iap = CGCNN_IAP(modelpath,datapath)

# can attach this calculator to an atoms object
# for use in other ASE routines (MD, minimization, etc.)
kwargs = {'cgcnn_iap': iap}
cgcnncalc = CGCNNCalculator(**kwargs)
dummy.set_calculator(cgcnncalc)

start =time.time()
numteststructs = 1
for i in range(numteststructs):
    dummy.rattle(stdev=0.2)
    energy = iap.predict(dummy)
    print(i,energy[0][0])
end = time.time()
print('Elapsed: %.6f s'%(end-start))
print('Elapsed per structure: %.6f s'%((end-start)/numteststructs))

start =time.time()
numteststructs = 1 
for i in range(numteststructs):
    dummy.rattle(stdev=0.2)
    energy = dummy.get_potential_energy()
    print(i,energy[0][0])
end = time.time()
print('Elapsed: %.6f s'%(end-start))
print('Elapsed per structure: %.6f s'%((end-start)/numteststructs))

# still intractably long
#start =time.time()
#forces = dummy._calc.calculate_numerical_forces(dummy)
#end = time.time()
#print('Elapsed: %.6f s'%(end-start))

# gradient free cell optimization...
# TODO why doesn't this work anymore...
posoptim = False
if posoptim:
    dummyorig = deepcopy(dummy)                                               
    dummyorig.set_calculator(cgcnncalc)                                            
                                                                              
    print(dir(ase.optimize.sciopt))                                           
                                                                              
    print(dummy) 
    opt = ase.optimize.sciopt.SciPyFmin(dummy,trajectory='opt.traj',logfile='opt.log')
    x = opt.run(steps=10)
    dummy.set_positions(x[0][:-6].reshape(-1,3))
    dummy.set_cell(x[0][-6:])

    finale = dummy.get_potential_energy()                                     
    orige = dummyorig.get_potential_energy()                                  
                                                                              
    print("orig positions:")                                                  
    print(dummyorig.get_positions())                                          
    print("final positions:")                                                 
    print(dummy.get_positions())                                              
                                                                              
    print("deltapositions:")                                                  
    print(dummy.get_positions()-dummyorig.get_positions())                    
    print("delta UC:")                                                        
    print(dummy.get_cell_lengths_and_angles()-dummyorig.get_cell_lengths_and_angles())
    print("final E, initial E, delta E:")                                     
    print(finale, orige, finale-orige)
