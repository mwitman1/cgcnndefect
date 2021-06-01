#! /usr/bin/env python3

from pprint import pprint
import json

with open('atom_init.json','r') as f:
    text = f.read()
aj = json.loads(text)
#pprint(aj)
for key in aj.keys():
    print(key, len(aj[key]), list(aj[key]))
last = [aj[key][-1] for key in aj.keys()]
print(last)
