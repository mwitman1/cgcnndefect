#! /usr/local/env python3

import torch

def aterm(Zi, Zj):

    return 0.46850e-10/(torch.pow(Zi,0.23) + torch.pow(Zj,0.23))

def phiterm(x):
    
    return  0.18175*torch.exp(-3.19980*x) + 0.50986*torch.exp(-0.94229*x) +\
            0.28022*torch.exp(-0.40290*x) + 0.02817*torch.exp(-0.20162*x)

def energyZBL(Zi,Zj,rij):
    """
    Computes the Ziegler-Biersack-Littmark (ZBL) term
    https://lammps.sandia.gov/doc/pair_zbl.html
    Which is useful as a "univesal" nuclear repulsive term fit to previous data

    Zi : torch.LongTensor of shape (N, M)
    Zj : torch.LongTensor of shape (N, M)
    rij : torch.Tensor of shape (N,M) , units are Angstrom

    returns : torch.Tensor of shape (N,M)
        The ZBL pair energy [eV]
    """

    # eps0 = 8.85418781762039e-12 [C^2 / (N m^2)]
    # e^2 = 1.60217662e-19^2 = 2.5669699e-38 [C]
    # 1.60217653e-19 [J/ev]
    
    #prefactor = 2.5669699e-38/(4*3.1415926535*8.85418781762039e-12)/1.60217653e-19
    prefactor = 1.4399646e-9
    rij_m=rij*1e-10
    #return prefactor * Zi * Zj * phiterm(rij/aterm(Zi,Zj)) / rij
    return prefactor * Zi * Zj * phiterm(rij_m/aterm(Zi,Zj)) / rij_m
