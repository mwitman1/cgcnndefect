#! /usr/bin/env python3

import json
import numpy as np
from copy import deepcopy
from matminer.featurizers.composition import ElementProperty
import pandas as pd

ATOMIC_NUMBER = [                                                               
    "ZERO", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",   
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",   
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",     
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",      
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",      
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",     
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",      
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",      
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",     
    "Ds", "Rg", "Cn", "Uut", "Uuq", "Uup", "Uuh", "Uuo"]

with open('atom_init.json.orig','r') as f:
    s = json.loads(f.read())

# Type 1:
# modify the original json to have one more entry for element "0" (defect)
# the one-hot feature feactor for each element gets one more dimension
# with 1 for "0" and 0 for "1" ... "92"
s1 = deepcopy(s)
s1['0']=list([0 for _ in range(len(s1['1']))])
for key in s1.keys():
    if key == '0':
        s1[key].append(1)
    else:
        s1[key].append(0)
with open('atom_init.json.defect','w') as f:
    f.write(json.dumps(s1,sort_keys=True))

# Type 2:
# no elemental data, can only be used if model is incorporating node-specific,
# local features from a different data source
s2 = deepcopy(s)
for key in s2.keys():
    s2[key] = []
with open('atom_init.json.none','w') as f:
    f.write(json.dumps(s2,sort_keys=True))

# Type 3: TODO
# each element now has TWO orig_atom_fea vectors
# orig_atom_fea_contin and orig_atom_fea_disc
df_elem = pd.DataFrame()
df_elem["composition"] = ATOMIC_NUMBER[1:101]
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df_elem = ep_feat.featurize_dataframe(df_elem, col_id="composition")
print(df_elem)
 

with open('atom_init.json.defect_multi','w') as f:
    pass

