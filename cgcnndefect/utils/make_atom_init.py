#! /usr/bin/env python3

import json
import numpy as np
from copy import deepcopy


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

# Type 2: TODO
# possible elements are subset of periodic table, only feature is element id


