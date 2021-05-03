#! /usr/bin/env python3

from ase.calculators.calculator import FileIOCalculator
from ase.utils import workdir
from subprocess import check_output
from ase.io import read,write
import ase.optimize.sciopt

import time
import numpy as np
from copy import deepcopy
import sys

# TODO, we need a custom version of CGCNN where the module can be imported
# and an energy predicted directly from the atoms object, not going through
# the command line and doing all the unnecessary stuff in cgcnn/predict.py


class CGCNN(FileIOCalculator):
    """Class for doing a CGCNN energy calculation from a pre-trained model
    """

    implemented_properties = ['energy']

    def __init__(self,restart=None,ignore_bad_restart_file=False,
                 label='cgcnn', atoms=None, command=None, **kwargs):

        FileIOCalculator.__init__(self,restart,ignore_bad_restart_file,
                                  label, atoms, command,**kwargs)

    def write_input(self,atoms,properties,system_changes):
        write("data/tmpcalculator/tmp.cif",atoms)

    def read(self,label):
        with open("data/tmpcalculator/all_results.csv", "r") as f:
            self.results = {'energy': float(f.readlines()[0].strip().split(',')[-1])}

    def read_results(self):
        with open("data/tmpcalculator/all_results.csv", "r") as f:
            self.results = {'energy': float(f.readlines()[0].strip().split(',')[-1])}
            print(self.atoms)
            print(self.results)

#atoms = read("./data/AlLiMgSnZnHEAOptTraj/AlLiMgSnZn_s233_tag233-5.cif")
testfname="/Users/mwitman/Applications/SSHEAGen/AlLiMgSnZn-xml-files/AlLiMgSnZnHEAOptTraj/AlLiMgSnZn_s106_tag106-5.cif"
atoms = read(testfname)
trueforces=np.loadtxt(testfname[:-4]+"_forces.csv",delimiter=',')

#atoms = ase.build.make_supercell(atoms,3*np.eye(3))

calc = CGCNN(command="python3 predict.py model_best.pth.tar data/tmpcalculator/ --resultdir data/tmpcalculator")
atoms.set_calculator(calc)

posoptim = False
if posoptim:
    atomsorig = deepcopy(atoms)
    atomsorig.set_calculator(calc)

    print(dir(ase.optimize.sciopt))

    print(atoms)
    opt = ase.optimize.sciopt.SciPyFmin(atoms,trajectory='opt.traj',logfile='opt.log')

    x = opt.run(steps=10)
    atoms.set_positions(x[0][:-6].reshape(-1,3))
    atoms.set_cell(x[0][-6:])


    finale = atoms.get_potential_energy()
    orige = atomsorig.get_potential_energy()

    print("orig positions:")
    print(atomsorig.get_positions())
    print("final positions:")
    print(atoms.get_positions())

    print("deltapositions:")
    print(atoms.get_positions()-atomsorig.get_positions())
    print("delta UC:")
    print(atoms.get_cell_lengths_and_angles()-atomsorig.get_cell_lengths_and_angles())
    print("final E, initial E, delta E:")
    print(finale, orige, finale-orige)

    write("data/tmpcalculator/tmpopt.cif",atoms)

numforces = True
if numforces:

    #for disp in [0.001,0.01,0.02,0.05,0.1]:
    for disp in [0.2,0.5,1.0]:
        start = time.time()
        forces = atoms._calc.calculate_numerical_forces(atoms,d=disp)
        end = time.time()
        elapsed = end-start
        
        with open("./testforces/test_numerical_forces_timing.txt","w") as f:
            f.write("Numerical forces calc time elapsed: %.1f s"%elapsed)
        
        np.savetxt("./testforces/test_numerical_forces_disp%f.txt"%disp,forces)
        np.savetxt("./testforces/true_forces.txt",trueforces)
        np.savetxt("./testforces/test_numerical_forces_diff_disp%f.txt"%disp,forces-trueforces)


