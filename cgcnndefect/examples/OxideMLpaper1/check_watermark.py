#! /usr/bin/env python3

from watermark import watermark
import pymatgen,ase,torch,numpy,sklearn
import os

osinfo=str(watermark())
libs=str(watermark(packages="pymatgen,ase,torch,numpy,sklearn,watermark"))

print('Your watermark:')
print(osinfo)
print(libs)
# for some reason watermark just reads the 0.0 from sklearn 
# and not the proper version for scikit-learn
print("".join(os.popen('pip3 show scikit-learn').readlines()[0:2]))

#with open('tested_watermark.txt','w') as f:
#    f.writelines(osinfo)
#    f.writelines(libs)
#    f.writelines("".join(os.popen('pip3 show scikit-learn').readlines()[0:2]))

print('Tested watermark:')
with open('tested_watermark.txt','r') as f:
    print(f.read())
