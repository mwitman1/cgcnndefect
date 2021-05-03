#! /usr/bin/env python3

from amp.descriptor.gaussian import calculate_G2
from ase.neighborlist import NeighborList
from ase.io import read,write
from ase.build import bulk, make_supercell
from ase.data import atomic_numbers
from amp.descriptor.cutoffs import Cosine, dict2cutoff
import numpy as np

image = bulk('Cu', 'fcc', a=3.6)
image = make_supercell(image, 10*np.eye(3))

chemical_symbols = np.array(image.get_chemical_symbols())
cutoff=8
nbrlst = NeighborList([cutoff]*len(image))
nbrlst.update(image)

nl = [nbrlst.get_neighbors(index) for index in range(len(image))]

for atom in image:

    Ri = atom.position

    symbol = atom.symbol
    index = atom.index
    neighborindices, neighboroffsets = nl[index]
    neighborsymbols = chemical_symbols[neighborindices]
    neighborpositions = (image.positions[neighborindices] +
                             np.dot(neighboroffsets,
                                    image.get_cell()))
    
    neighbornumbers = [atomic_numbers[_] for _ in neighborsymbols]

    print(neighbornumbers)
    print(neighborsymbols)
    print(neighborpositions)

    ridge = calculate_G2(neighbornumbers, neighborsymbols,
                         neighborpositions, 'Cu', 1.0,
                         0.5, cutoff, Ri, False)



