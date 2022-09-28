#! /usr/bin/env python3

from ase.io import read,write
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.special import sph_harm
from math import comb
from pprint import pprint
import numpy as np
import sys
import torch
import torch.nn as nn
from typing import Tuple, List
import time
from copy import deepcopy


def polar_coords(cart_vec):

    xy = cart_vec[:,0]**2 + cart_vec[:,1]**2

    #angles = np.empty((cart_vec.shape[0],2))

    r = np.sqrt(xy + cart_vec[:,2]**2) 

    # polar/inclination angle
    #angles[:,0] = np.arctan2(np.sqrt(xy), cart_vec[:,2])
    polar = np.arctan2(np.sqrt(xy), cart_vec[:,2])

    # azimuthal angle
    #angles[:,1] = np.arctan2(cart_vec[:,1], cart_vec[:,0])
    azimuth = np.arctan2(cart_vec[:,1], cart_vec[:,0])

    return r, polar, azimuth

def fcut(r, rcut):
    
    fcutr = np.exp(-np.power(r,2)/((rcut-r)*(rcut+r)))
    fcutr[np.where(r>rcut)[0]]=0
    
    return fcutr

def compute_rhok(r, k, K, rcut, gamma=0.9):

    rhok = comb(K-1,k)*np.exp(-gamma*r)**k*(1-np.exp(-gamma*r))**(K-1-k)*\
           fcut(r,rcut) 

    return rhok
           

def get_harmonics_fea(structure, all_nbrs, K, rcut, njmax=0):
    """
    May need to optimize/vectorize as much of this as possible later on

    N - number of atoms in crystal
    nj - number of neighbors of atom i
    njmax - if positive, capped number of max neighbors of atom i

    returns 
        gs_fea : list of size N of np.arrays with shape(
    """

    gs_fea, gp_fea, gd_fea = [], [], []
    nbr_dist, nbr_vec = [], []
    nbr_angles = []
    for i, nbrs in enumerate(all_nbrs):
        # Using sph_harm basis set, no longer need to artifically cap the 
        # number of interactions (assigned bonds) to a fixed max value
        # within the cutoff, so no need to sort
        # However, zero-padding up to max neighbors allows vectorization
        # of all operations despite the non-constant num of neighbors between atoms
        # Therefore, a present njmax is being used, and program will automatically
        # exit if more neighbors exist in the cutoff than 
        if len(nbrs) >= njmax and njmax != 0:
            raise ValueError("Warning! You're max num of possible neighbors is not sufficient to capture all neighbors within the requested cutoff!")
            #print("Warning! More neighs in cutoff than njmax")

        # variable number of neighbors w/in site for each site
        #nbr_fea_idx.append([nbr[2] for nbr in nbrs])
         
        # Need to be especially careful to use Pymatgen Periodic neighbor
        # to get correct distance vector
        icoords = structure[i].coords
        nbr_vec = np.array([nbr[0].coords-icoords for nbr in nbrs])
        #if i == 0:
        #    print(np.sort(np.linalg.norm(nbr_vec,axis=1)))
        r, polar, azimuth = polar_coords(nbr_vec)
        
        # nbr_gx_ ... with shape (nj,)
        # NOTE: l,m,theta,phi (note opposite of the usual theta/phi convention)
        # NOTE: range of theta/phi accepted by sph_harm not (-pi,pi)
        #       i.e., output of arctan, needs digging
        nbr_gs_0_0  = np.real(sph_harm( 0, 0, azimuth, polar)) 
        
        nbr_gp_1_n1 = np.real(sph_harm(-1, 1, azimuth, polar))
        nbr_gp_1_0  = np.real(sph_harm( 0, 1, azimuth, polar))
        nbr_gp_1_1  = np.real(sph_harm( 1, 1, azimuth, polar))

        nbr_gd_2_n2 = np.real(sph_harm(-2, 2, azimuth, polar))
        nbr_gd_2_n1 = np.real(sph_harm(-1, 2, azimuth, polar))
        nbr_gd_2_0  = np.real(sph_harm( 0, 2, azimuth, polar))
        nbr_gd_2_1  = np.real(sph_harm( 1, 2, azimuth, polar))
        nbr_gd_2_2  = np.real(sph_harm( 2, 2, azimuth, polar))

        # nbr_rhok with shape(K, nj)
        nbr_rhok = np.array([compute_rhok(r, k, K, rcut) for k in range(K)])

        # nbr_gs_fea with shape(nj, K)
        nbr_gs_fea=(nbr_gs_0_0*nbr_rhok).T[..., np.newaxis]


        # nbr_gp_fea with shape(nj, K, 3)
        nbr_gp_fea=np.stack(((nbr_gp_1_n1*nbr_rhok).T,
                             (nbr_gp_1_0*nbr_rhok).T,
                             (nbr_gp_1_1*nbr_rhok).T),axis=2)


        
        # nbr_gp_fea with shape(nj, K, 5)
        nbr_gd_fea=np.stack(((nbr_gd_2_n2*nbr_rhok).T,
                             (nbr_gd_2_n1*nbr_rhok).T,
                             (nbr_gd_2_0*nbr_rhok).T,
                             (nbr_gd_2_1*nbr_rhok).T,
                             (nbr_gd_2_2*nbr_rhok).T),axis=2)


        # Checks
        #--------------------------------------------------------------------

        # 1. computed r_ij from scipy sph_harm same as from pymatgen nbr list
        #dists = [nbr[1] for nbr in nbrs]
        #nbr_dist.append(dists)
        #assert(np.allclose(np.array(dists),r))

        # 2. vectorized calc same as manual calc
        # For gs 
        #assert(np.isclose(nbr_gs_fea[0,0],nbr_gs_0_0[0]*nbr_rhok[0,0]))
        #assert(np.isclose(nbr_gs_fea[0,1],nbr_gs_0_0[0]*nbr_rhok[1,0]))
        #assert(np.isclose(nbr_gs_fea[0,2],nbr_gs_0_0[0]*nbr_rhok[2,0]))
        #assert(np.isclose(nbr_gs_fea[0,3],nbr_gs_0_0[0]*nbr_rhok[3,0]))

        #assert(np.isclose(nbr_gs_fea[1,0],nbr_gs_0_0[1]*nbr_rhok[0,1]))
        #assert(np.isclose(nbr_gs_fea[1,1],nbr_gs_0_0[1]*nbr_rhok[1,1]))
        #assert(np.isclose(nbr_gs_fea[1,2],nbr_gs_0_0[1]*nbr_rhok[2,1]))
        #assert(np.isclose(nbr_gs_fea[1,3],nbr_gs_0_0[1]*nbr_rhok[3,1]))

 
        ## For Gp 
        #assert(np.isclose(nbr_gp_fea[1,3,0],nbr_gp_1_n1[1]*nbr_rhok[3,1]))
        #assert(np.isclose(nbr_gp_fea[1,3,1],nbr_gp_1_0[1]*nbr_rhok[3,1]))
        #assert(np.isclose(nbr_gp_fea[1,3,2],nbr_gp_1_1[1]*nbr_rhok[3,1]))

        ## For Gd
        #assert(np.isclose(nbr_gd_fea[-1,2,0],nbr_gd_2_n2[-1]*nbr_rhok[2,-1]))
        #assert(np.isclose(nbr_gd_fea[-1,2,1],nbr_gd_2_n1[-1]*nbr_rhok[2,-1]))
        #assert(np.isclose(nbr_gd_fea[-1,2,2],nbr_gd_2_0[-1]*nbr_rhok[2,-1]))
        #assert(np.isclose(nbr_gd_fea[-1,2,3],nbr_gd_2_1[-1]*nbr_rhok[2,-1]))
        #assert(np.isclose(nbr_gd_fea[-1,1,4],nbr_gd_2_2[-1]*nbr_rhok[1,-1]))
        #--------------------------------------------------------------------

        if njmax > 0:
            # padding 0's up to njmax so that can be vectorized
            nbr_gs_fea_pad=np.zeros((njmax-len(nbrs),K,1))
            nbr_gp_fea_pad=np.zeros((njmax-len(nbrs),K,3))
            nbr_gd_fea_pad=np.zeros((njmax-len(nbrs),K,5))

            gs_fea.append(torch.Tensor(np.concatenate((nbr_gs_fea,nbr_gs_fea_pad),axis=0)))
            gp_fea.append(torch.Tensor(np.concatenate((nbr_gp_fea,nbr_gp_fea_pad),axis=0)))
            gd_fea.append(torch.Tensor(np.concatenate((nbr_gd_fea,nbr_gd_fea_pad),axis=0)))
        else:
            gs_fea.append(torch.Tensor(nbr_gs_fea))
            gp_fea.append(torch.Tensor(nbr_gp_fea))
            gd_fea.append(torch.Tensor(nbr_gd_fea))

    # returning lists of tensors here keeps things flexible for later on,
    # regardless of njmax specification
    return gs_fea, gp_fea, gd_fea


class Residual(nn.Module):
    def __init__(self,fea_len_in, fea_len_out):
        super(Residual, self).__init__()
        self.linear1 = nn.Linear(fea_len_in,fea_len_out)
        self.silu1 = nn.SiLU()
        #self.bn1 = nn.BatchNorm1d(fea_len_out)

    def forward(self,x):
        return x + self.linear1(self.silu1(x))

class ResMLP(nn.Module):
    def __init__(self,fea_len_in, fea_len_out):
        super(ResMLP, self).__init__()
        self.residual = Residual(fea_len_in,fea_len_in)
        self.silu1 = nn.SiLU()
        self.linear1 = nn.Linear(fea_len_in,fea_len_out)
        self.bn1 = nn.BatchNorm1d(fea_len_in)

    def forward(self,x):
        return self.linear1(self.silu1(self.residual(self.bn1(x))))

class InvertedLinear(nn.Module):
    def __init__(self,fea_len_in, fea_len_out):
        super(InvertedLinear, self).__init__()
        self.linear1 = nn.Linear(fea_len_in,fea_len_out)
        self.silu1 = nn.SiLU()

    def forward(self,x):
        return self.linear1(self.silu1(x))


class SpookyLocalBlock(nn.Module):

    def __init__(self, atom_fea_len, K):
        super(SpookyLocalBlock, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.F = atom_fea_len
        self.K = K

        self.Gs = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,K)))
        self.Gp = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,K)))
        self.Gd = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,K)))

        self.P1 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))
        self.P2 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))
        self.D1 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))
        self.D2 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))

        self.resmlp_c = ResMLP(atom_fea_len,atom_fea_len)
        self.resmlp_s = ResMLP(atom_fea_len,atom_fea_len)
        self.resmlp_p = ResMLP(atom_fea_len,atom_fea_len)
        self.resmlp_d = ResMLP(atom_fea_len,atom_fea_len)

    def forward(self, atom_fea : torch.Tensor,
                      nbr_fea_idx : List[torch.LongTensor],
                      gs : List[torch.Tensor],
                      gp : List[torch.Tensor],
                      gd : List[torch.Tensor]):
        """
        Test forward vectorized using 0 padding for a pre-selected nj max
        """
        # TODO
        # shape (N, F)
        s_fea = self.resmlp_s(atom_fea)
        # shape (N, F)
        p_fea = self.resmlp_p(atom_fea)
        # shape (N, F)
        d_fea = self.resmlp_d(atom_fea)

        # OPTION 2: non-vectorized version
        # -> nonvectorized version, if nj is diff for each atomic environment
        all_s = []
        all_p = []
        all_d = []
        for i in range(len(gs)):
            # shape (nj, atom_fea_len, 1)
            s_env = torch.matmul(self.Gs, gs[i]) # note always 0 for padding above njmax
            # shape (nj, atom_fea_len)
            nbr_s_fea = s_fea.index_select(0, nbr_fea_idx[i])
            # shape (atom_fea_len) 
            si = torch.sum(nbr_s_fea.unsqueeze(-1)*s_env,0).squeeze()
            all_s.append(si)
            #print('S vars')
            #print(s_fea.shape, s_env.shape, nbr_s_fea.shape)
            #print(si.shape)

            # shape (nj, atom_fea_len, 3)
            p_env = torch.matmul(self.Gp, gp[i])
            # shape (nj, atom_fea_len)
            nbr_p_fea = p_fea.index_select(0, nbr_fea_idx[i])
            # shape (atom_fea_len, 3) 
            pi = torch.sum(nbr_p_fea.unsqueeze(-1).expand(\
                 p_env.shape[0], p_env.shape[1], p_env.shape[2])*p_env,0)
            all_p.append(pi)
            #print('P vars')
            #print(p_fea.shape, p_env.shape, nbr_p_fea.shape)
            #print(pi.shape)

            # shape (nj, atom_fea_len, 5)
            d_env = torch.matmul(self.Gd, gd[i])
            # shape (nj, atom_fea_len)
            nbr_d_fea = d_fea.index_select(0, nbr_fea_idx[i])
            # shape (atom_fea_len, 5) 
            di = torch.sum(nbr_d_fea.unsqueeze(-1).expand(\
                 d_env.shape[0], d_env.shape[1], d_env.shape[2])*d_env,0)
            all_d.append(di)
            #print('D vars')
            #print(d_fea.shape, d_env.shape, nbr_d_fea.shape)
            #print(di.shape)

        # shape (N, 3, atom_fea_len)
        all_p = torch.transpose(torch.stack(all_p),1,2)

        # shape (N, 5, atom_fea_len)
        all_d = torch.transpose(torch.stack(all_d),1,2)

        # shape(N, atom_fea_len)
        final_c = self.resmlp_c(atom_fea)

        # shape(N, atom_fea_len)
        final_s = torch.stack(all_s)

        # inner prod of eq(12) dimensionality doesn't seem to work out
        # < P1 p , P2 p > \in R ?? = Tr( (P2 p)^T \dot (P1 p) )
        # shape (N, 3, atom_fea_len)
        p1term = torch.matmul(all_p, self.P1)
        # shape (N, 3, atom_fea_len)
        p2term = torch.matmul(all_p, self.P2)
        # shape (N, atom_fea_len) via broadcasting in dim1
        final_p =\
            torch.sum(\
             torch.diagonal(\
                    torch.matmul(torch.transpose(p2term,1,2),p1term), 
                    dim1=-2, dim2=-1),
            dim=1).unsqueeze(-1).expand(atom_fea.shape[0], self.atom_fea_len)


        # same issue for < D1 d, D2 d >
        # shape (N, 5, atom_fea_len)
        d1term = torch.matmul(all_d, self.D1)
        # shape (N, 5, atom_fea_len)
        d2term = torch.matmul(all_d, self.D2)
        # shape (N, atom_fea_len) via broadcasting in dim1
        final_d =\
            torch.sum(
             torch.diagonal(
                torch.matmul(torch.transpose(d2term,1,2),d1term),
                dim1=-2, dim2=-1),
            dim=1).unsqueeze(-1).expand(atom_fea.shape[0], self.atom_fea_len)

        l = final_c +\
            final_s +\
            final_p +\
            final_d
            
        return l

class SpookyLocalBlockVectorized(nn.Module):

    def __init__(self, atom_fea_len, K, njmax):
        super(SpookyLocalBlockVectorized, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.F = atom_fea_len
        self.K = K
        self.njmax = njmax

        self.Gs = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,K)))
        self.Gp = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,K)))
        self.Gd = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,K)))

        self.P1 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))
        self.P2 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))
        self.D1 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))
        self.D2 = nn.init.uniform_(nn.Parameter(torch.Tensor(atom_fea_len,atom_fea_len)))

        self.resmlp_c = ResMLP(atom_fea_len,atom_fea_len)
        self.resmlp_s = ResMLP(atom_fea_len,atom_fea_len)
        self.resmlp_p = ResMLP(atom_fea_len,atom_fea_len)
        self.resmlp_d = ResMLP(atom_fea_len,atom_fea_len)

    def forward(self, atom_fea : torch.Tensor,
                      nbr_fea_idx : torch.LongTensor,
                      gs : torch.Tensor,
                      gp : torch.Tensor,
                      gd : torch.Tensor):

        """
        N : number of atoms in batch
        njmax : max number of neighbors to an atom for sph_harm
        F : atom feacture vector length
        K : order of polynomials

        atom_fea : torch.Tensor of shape (N, F)
            -> init feature vectors of all atoms in the batch
        nbr_fea_idx : torch.LongTensor of shape (N*njmax)
            -> indices of all neighbors W.R.T. the batch (NOT the crystal)
            -> note that the padding of neighbors beyond the cutoff (up to njmax) stores
               a dummy index corresponding to that cyrstal's first atom's index
               in the batch. Ultimately doesn't matter b/c gs, gp, and gd
               values are padded with 0s for any such nbs beyond the cutoff 
               up to njmax
        gs : torch.Tensor of shape(N, nj, K, 1)
        gp : torch.Tensor of shape(N, nj, K, 3)
        gd : torch.Tensor of shape(N, nj, K, 5)

        returns
        l : torch.Tensor of shape (N,F)

        """

        # shape(N, atom_fea_len)
        final_c = self.resmlp_c(atom_fea)
        # shape (N, F)
        s_fea = self.resmlp_s(atom_fea)
        # shape (N, F)
        p_fea = self.resmlp_p(atom_fea)
        # shape (N, F)
        d_fea = self.resmlp_d(atom_fea)

       
        # OPTION 1: vectorized version
        # -> when each atom env has padded 0s to make up to exactly njmax neighbors
 
        #print(self.Gs.expand(len(gs),gs[0].shape[0],self.Gs.shape[0],self.Gs.shape[1]).shape)
        #print(torch.stack(gs,dim=0).shape, len(gs))
        # shape (N, njmax, F, 1)
        s_env = torch.matmul(self.Gs.expand(len(gs), # N
                                            self.njmax, # njmax
                                            self.F, # F
                                            self.K), # K 
                             gs) # (N, njmax, K, 1)
        #print(s_env.shape)

        # shape (N, njmax, F)
        #nbr_s_fea = torch.stack([s_fea.index_select(0,nbr_fea_idx[i])\
        #                         for i in range(len(nbr_fea_idx))])
        nbr_s_fea =\
            s_fea.index_select(0,nbr_fea_idx).reshape(\
                len(gs), self.njmax, self.F)
        #assert torch.allclose(nbr_s_fea, nbr_s_fea_nonvec)
        #print(nbr_s_fea.shape)

        # shape (N,atom_fea)
        final_s = torch.sum(nbr_s_fea.unsqueeze(-1)*s_env,dim=1).squeeze()
        #print(final_s_vectorized.shape)
        

        # shape (N, nj, F, 3)
        p_env = torch.matmul(self.Gp.expand(len(gs), # N
                                            self.njmax, # njmax
                                            self.F, # F
                                            self.K), # K
                             gp) # (N, njmax, K, 3)
        #                     torch.stack(gp,dim=0)) # (N, njmax, K, 3)
        #print(p_env.shape)

        # shape (N, nj, F)
        #nbr_p_fea = torch.stack([p_fea.index_select(0, nbr_fea_idx[i])\
        #                        for i in range(len(nbr_fea_idx))])
        nbr_p_fea =\
            p_fea.index_select(0,nbr_fea_idx).reshape(\
                len(gp), self.njmax, self.F)
        #print(nbr_p_fea.shape)

        # shape (N, F, 3)  
        all_p = torch.sum(nbr_p_fea.unsqueeze(-1).expand(\
                     p_env.shape[0], p_env.shape[1], self.F, 3)*p_env,dim=1)
        #print(all_p.shape)

        # shape (N, 3, F)  
        #all_p = torch.transpose(all_p,1,2)

        # inner prod of eq(12) dimensionality doesn't seem to work out
        # < P1 p , P2 p > \in R ?? = Tr( (P2 p)^T \dot (P1 p) )
        # shape (N, 3, F)
        #p1term = torch.matmul(all_p, self.P1)
        # shape (N, 3, F)
        #p2term = torch.matmul(all_p, self.P2)
        # shape (N, atom_fea_len) via broadcasting in dim1
        #final_p =\
        #    torch.sum(\
        #     torch.diagonal(\
        #            torch.matmul(torch.transpose(p2term,1,2),p1term), 
        #            dim1=-2, dim2=-1),
        #    dim=1).unsqueeze(-1).expand(atom_fea.shape[0], self.atom_fea_len)

        # shape (N, F)
        final_p = torch.sum(torch.matmul(self.P1,all_p)*torch.matmul(self.P2,all_p),dim=2)


        # shape (N, nj, F, 5)
        d_env = torch.matmul(self.Gd.expand(len(gd), # N
                                            self.njmax, # njmax
                                            self.F, # F
                                            self.K), # K
                             gd) # (N, njmax, K, 5)
        #                     torch.stack(gd,dim=0)) # (N, njmax, K, 5)

        # shape (N, nj, F)
        #nbr_d_fea = torch.stack([d_fea.index_select(0, nbr_fea_idx[i])\
        #                        for i in range(len(nbr_fea_idx))])
        nbr_d_fea =\
            d_fea.index_select(0,nbr_fea_idx).reshape(\
                len(gd), self.njmax, self.F)

        # shape (N, F, 5) 
        all_d = torch.sum(nbr_d_fea.unsqueeze(-1).expand(\
                     d_env.shape[0], d_env.shape[1], self.F, 5)*d_env,dim=1)

        # shape (N, 5, F) 
        #all_d = torch.transpose(all_d,1,2)

        # same issue for < D1 d, D2 d >
        # shape (N, 5, atom_fea_len)
        #d1term = torch.matmul(all_d, self.D1)
        # shape (N, 5, atom_fea_len)
        #d2term = torch.matmul(all_d, self.D2)
        # shape (N, atom_fea_len) via broadcasting in dim1
        #final_d =\
        #    torch.sum(
        #     torch.diagonal(
        #        torch.matmul(torch.transpose(d2term,1,2),d1term),
        #        dim1=-2, dim2=-1),
        #    dim=1).unsqueeze(-1).expand(atom_fea.shape[0], self.atom_fea_len)

        # shape (N, F)
        final_d = torch.sum(torch.matmul(self.D1,all_d)*torch.matmul(self.D2,all_d),dim=2)


        #print(final_c)
        #print(final_s)
        #print(final_p)
        #print(final_d)

        # TODO: BOTH APPROACHES MUST GIVE SAME RESULT ! 
        # Why are the largest differences ~ 1e-6 
        #final_s_nonvec = torch.stack(all_s_nonvec)
        #print(final_s - final_s_nonvec)
        #print(torch.where(torch.abs(final_s - final_s_nonvec)<1e-9))
        #print('Max diff s: ', torch.max(final_s - final_s_nonvec))
        #assert torch.allclose(final_s, final_s_nonvec)

        l = final_c +\
            final_s +\
            final_p +\
            final_d
            
        return l

class SpookyConv(nn.Module):
    def __init__(self, atom_fea_len, K):
        super(SpookyConv, self).__init__()

        self.res1 = Residual(atom_fea_len, atom_fea_len)
        self.res2 = Residual(atom_fea_len, atom_fea_len)
        self.spookylocal = SpookyLocalBlock(atom_fea_len, K)
        

    def forward(self, atom_fea, nbr_fea_idx, 
                      gs_fea, gp_fea, gd_fea):

        intermed_atom_fea = self.res1(atom_fea)
        local = self.spookylocal(intermed_atom_fea, nbr_fea_idx,
                                 gs_fea, gp_fea, gd_fea)
        atom_fea = self.res2(intermed_atom_fea + local)
        
        return atom_fea

class SpookyConvVectorized(nn.Module):
    def __init__(self, atom_fea_len, K, njmax):
        super(SpookyConvVectorized, self).__init__()

        self.res1 = Residual(atom_fea_len, atom_fea_len)
        self.res2 = Residual(atom_fea_len, atom_fea_len)
        self.spookylocal = SpookyLocalBlockVectorized(atom_fea_len, K, njmax)

    def forward(self, atom_fea, nbr_fea_idx, 
                      gs_fea, gp_fea, gd_fea):

        intermed_atom_fea = self.res1(atom_fea)
        local = self.spookylocal(intermed_atom_fea, nbr_fea_idx,
                                 gs_fea, gp_fea, gd_fea)
        atom_fea = self.res2(intermed_atom_fea + local)
        
        return atom_fea

class SpookyModel(nn.Module):

    def __init__(self, orig_atom_fea_len, atom_fea_len=64, n_conv=3, 
                       h_fea_len=128, n_h=1, K=4, all_elems = [0], global_fea_len=0,
                       pooltype='all'):
        super(SpookyModel, self).__init__()

        self.n_h = n_h
        self.pooltype = pooltype

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
       
        # Note, njmax is used as switch for vectorized version (significantly faster)
        # When njmax > 0,  assumes 0 padding up to njmax in gs, gp, gd
        # which will be tensors of shape (N, njmax, ...)
        # When njmax = 0, each atom has a variable number of neighboris in gs, gp, gd
        # which will be lists of len(N) of tensor with shape (nj, ...)
        self.convs = nn.ModuleList([SpookyConv(atom_fea_len=atom_fea_len,
                                               K = K)
                                   for _ in range(n_conv)])

        self.conv_to_fc = InvertedLinear(atom_fea_len+global_fea_len, h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([InvertedLinear(h_fea_len,h_fea_len)
                                      for _ in range(n_h-1)])

        self.fc_out = InvertedLinear(h_fea_len, 1)
    
    def forward(self, atom_fea, nbr_fea_idx, crystal_atom_idx,
                      gs_fea, gp_fea, gd_fea, global_fea):
        # embedding
        atom_fea = self.embedding(atom_fea)

        # pass throuch n_conv SpookyConv blocks
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea_idx, 
                                 gs_fea, gp_fea, gd_fea)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx, self.pooltype)

        crys_fea = self.conv_to_fc(torch.cat([crys_fea, global_fea],dim=1))

        if self.n_h > 1:
            for fc in self.fcs:
                crys_fea = fc(crys_fea)

        out = self.fc_out(crys_fea)

        return [out]

    def pooling(self, atom_fea : torch.Tensor,
                      crystal_atom_idx : List[torch.Tensor],
                      pooltype : str):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
          Must be a list of tensors since each idx_map is 
            tensor of different size (number of atoms in that crystal)
        """
        #assert torch.sum(torch.tensor([len(idx_map) for idx_map in\
        #    crystal_atom_idx])) == atom_fea.data.shape[0]


        # 1. normal pooling
        if pooltype == 'all':
            summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                          for idx_map in crystal_atom_idx]
        elif pooltype == '0':
            # 2. for defect, we are really only interested with the feature
            # vector of the node that would become the defect
            #print([idx_map[0] for idx_map in crystal_atom_idx])
            summed_fea = [torch.index_select(atom_fea,0,idx_map[0])\
                          for idx_map in crystal_atom_idx]
            #print(summed_fea)
        elif pooltype == 'none':
            return atom_fea
        else:
            raise ValueError("unallowed pooltype. must be in {'all','0'}")


        return torch.cat(summed_fea,dim=0)

class SpookyModelVectorized(nn.Module):

    def __init__(self, orig_atom_fea_len, atom_fea_len=64, n_conv=3, 
                       h_fea_len=128, n_h=1, K=4, all_elems = [0], global_fea_len=0,
                       njmax = 75, pooltype='all'):
        super(SpookyModelVectorized, self).__init__()

        self.n_h = n_h
        self.pooltype = pooltype

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
       
        # Note, njmax is used as switch for vectorized version (significantly faster)
        # When njmax > 0,  assumes 0 padding up to njmax in gs, gp, gd
        # which will be tensors of shape (N, njmax, ...)
        # When njmax = 0, each atom has a variable number of neighboris in gs, gp, gd
        # which will be lists of len(N) of tensor with shape (nj, ...)
        self.convs = nn.ModuleList([SpookyConvVectorized(atom_fea_len=atom_fea_len,
                                                         K = K,
                                                         njmax = njmax)\
                                   for _ in range(n_conv)])

        self.conv_to_fc = InvertedLinear(atom_fea_len+global_fea_len, h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([InvertedLinear(h_fea_len,h_fea_len)
                                      for _ in range(n_h-1)])

        self.fc_out = InvertedLinear(h_fea_len, 1)
    
    def forward(self, atom_fea, nbr_fea_idx, crystal_atom_idx,
                      gs_fea, gp_fea, gd_fea, global_fea):
        # embedding
        atom_fea = self.embedding(atom_fea)

        # pass throuch n_conv SpookyConv blocks
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea_idx, 
                                 gs_fea, gp_fea, gd_fea)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx, self.pooltype)

        crys_fea = self.conv_to_fc(torch.cat([crys_fea, global_fea],dim=1))

        if self.n_h > 1:
            for fc in self.fcs:
                crys_fea = fc(crys_fea)

        out = self.fc_out(crys_fea)

        return [out]

    def pooling(self, atom_fea : torch.Tensor,
                      crystal_atom_idx : List[torch.Tensor],
                      pooltype : str):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
          Must be a list of tensors since each idx_map is 
            tensor of different size (number of atoms in that crystal)
        """
        #assert torch.sum(torch.tensor([len(idx_map) for idx_map in\
        #    crystal_atom_idx])) == atom_fea.data.shape[0]


        # 1. normal pooling
        if pooltype == 'all':
            summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                          for idx_map in crystal_atom_idx]
        elif pooltype == '0':
            # 2. for defect, we are really only interested with the feature
            # vector of the node that would become the defect
            #print([idx_map[0] for idx_map in crystal_atom_idx])
            summed_fea = [torch.index_select(atom_fea,0,idx_map[0])\
                          for idx_map in crystal_atom_idx]
            #print(summed_fea)
        elif pooltype == 'none':
            return atom_fea
        else:
            raise ValueError("unallowed pooltype. must be in {'all','0'}")


        return torch.cat(summed_fea,dim=0)



def debug_featurization_single(crystal, cutoff, atom_fea_len, 
                               K, njmax, orig_atom_fea_len, seed):

    torch.manual_seed(seed)

    ########################################################################
    # example for one crystal
    ########################################################################
    t0 = time.time()
    all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
    print('Num neighs detected for atom 0: ', len(all_nbrs[0]))

    # some dummy atom feature vectors mapped to the input size of the conv layer 
    dummy_init_atom_fea = torch.Tensor(\
        [[crystal[i].specie.number for _ in range(atom_fea_len)]\
         for i in range(len(crystal))])
    print(dummy_init_atom_fea)

    # dummy atom features input to model (i.e. embedding layer)
    dummy_orig_atom_fea = torch.zeros((len(crystal),orig_atom_fea_len))
    for i in range(len(crystal)):
        #dummy_orig_atom_fea[i,crystal[i].specie.number] = 1
        dummy_orig_atom_fea[i,:] = crystal[i].specie.number
    print(dummy_orig_atom_fea)

    # indices of all neighbors    
    if njmax > 0:
        # nbr_fea_idx padded with 0s up to njmax for vectorized
        nbr_fea_idx = [torch.LongTensor(list(map(lambda x: x[2],nbr))\
                       + [0] * (njmax - len(nbr)))\
                       for nbr in all_nbrs]
    else:
        nbr_fea_idx = [torch.LongTensor(list(map(lambda x: x[2],nbr)))\
                       for nbr in all_nbrs]

    crystal_atom_idx = [torch.LongTensor(np.arange(len(crystal)))]
    t0e = time.time()

    # time the featurization
    t1 = time.time()
    gs_fea, gp_fea, gd_fea = get_harmonics_fea(crystal, all_nbrs, 
                                               K, cutoff, njmax)
    t1e = time.time()

    # test the isolated convolutional block
    if njmax > 0:
        gs_fea = torch.stack(gs_fea,dim=0)
        gp_fea = torch.stack(gp_fea,dim=0)
        gd_fea = torch.stack(gd_fea,dim=0)
        nbr_fea_idx = torch.cat(nbr_fea_idx,dim=0)

        l1 = SpookyLocalBlockVectorized(atom_fea_len, K, njmax)
        out_conv = l1.forward(dummy_init_atom_fea, nbr_fea_idx,  
                         gs_fea, gp_fea, gd_fea)
    else:
        l1 = SpookyLocalBlock(atom_fea_len, K)
        out_conv = l1.forward(dummy_init_atom_fea, nbr_fea_idx,  
                         gs_fea, gp_fea, gd_fea)

    if njmax > 0:
        # time entire model evaluation (for 1 crystal)
        model = SpookyModelVectorized(orig_atom_fea_len = orig_atom_fea_len,
                                      atom_fea_len = atom_fea_len,
                                      h_fea_len = 16,
                                      n_h=2,
                                      global_fea_len=2,
                                      njmax = njmax)
    else:
        model = SpookyModel(orig_atom_fea_len = orig_atom_fea_len,
                            atom_fea_len = atom_fea_len,
                            h_fea_len = 16,
                            n_h=2,
                            global_fea_len=2)
        

    model.eval()
    global_fea = torch.Tensor([[1,2]])

    t2 = time.time()
    out_model = model.forward(dummy_orig_atom_fea, nbr_fea_idx, crystal_atom_idx, 
                              gs_fea, gp_fea, gd_fea, global_fea)
    t2e = time.time()
    #print(model.convs[0].spookylocal.Gs)

    #print(out_model)
    print('Crystal processing time: ', t0e-t0)
    print('Featurization time: ', t1e-t1)
    print('Model time: ', t2e-t2)

    return gs_fea, gp_fea, gd_fea, out_conv, out_model

def debug_featurization_batch(crystal_batch, cutoff, atom_fea_len, 
                               K, njmax, orig_atom_fea_len, seed):

    torch.manual_seed(seed)

    ########################################################################
    # example for batch of crytals
    ########################################################################
    batch_orig_atom_fea, batch_atom_fea, batch_nbr_fea_idx, batch_crystal_atom_idx = [], [], [], []
    batch_gs_fea, batch_gp_fea, batch_gd_fea  = [], [], []
    batch_global_fea = []

    batch_size = len(crystal_batch)
    base_idx = 0
    for i,crystal in enumerate(crystal_batch):
        n_i = len(crystal_batch[i])
        all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)
        batch_crystal_atom_idx.append(torch.LongTensor(np.arange(n_i)+base_idx))
      
        # dummy atom features input to model (i.e. embedding layer)
        orig_atom_fea = torch.zeros((len(crystal),orig_atom_fea_len))
        for j in range(len(crystal)):
            #orig_atom_fea[j,crystal[j].specie.number] = 1
            orig_atom_fea[j,:] = crystal[j].specie.number
        batch_orig_atom_fea.append(orig_atom_fea)
 
        # some dummy atom feature vectors mapped to the input size of the conv layer 
        batch_atom_fea.append(
            torch.Tensor([[crystal[i].specie.number for _ in range(atom_fea_len)]\
                         for i in range(len(crystal))])
        )

        # indices of all neighbors    
        if njmax > 0:
            batch_nbr_fea_idx+=[base_idx + torch.LongTensor(list(map(lambda x: x[2],nbr))\
                                                            + [0]*(njmax-len(nbr)))\
                                for nbr in all_nbrs]
        else:
            batch_nbr_fea_idx+=[base_idx + torch.LongTensor(list(map(lambda x: x[2],nbr)))\
                                for nbr in all_nbrs]
        
        gs_fea, gp_fea, gd_fea = get_harmonics_fea(crystal, all_nbrs, 
                                                   K, cutoff, njmax)

        batch_gs_fea+=gs_fea
        batch_gp_fea+=gp_fea
        batch_gd_fea+=gd_fea

        base_idx += n_i

    batch_orig_atom_fea = torch.cat(batch_orig_atom_fea, dim=0)
    batch_atom_fea = torch.cat(batch_atom_fea, dim=0)
    batch_global_fea = torch.stack([torch.Tensor([1,2])\
                                     for _ in range(batch_size)], dim=0)


    # test the isolated convolutional block
    if njmax > 0:
        batch_gs_fea = torch.stack(batch_gs_fea,dim=0)
        batch_gp_fea = torch.stack(batch_gp_fea,dim=0)
        batch_gd_fea = torch.stack(batch_gd_fea,dim=0)
        batch_nbr_fea_idx = torch.cat(batch_nbr_fea_idx,dim=0)
        
        l1 = SpookyLocalBlockVectorized(atom_fea_len, K, njmax)
        out_conv = l1.forward(batch_atom_fea, batch_nbr_fea_idx, 
                              batch_gs_fea, batch_gp_fea, batch_gd_fea)
    else:
        l1 = SpookyLocalBlock(atom_fea_len, K)
        out_conv = l1.forward(batch_atom_fea, batch_nbr_fea_idx, 
                             batch_gs_fea, batch_gp_fea, batch_gd_fea)


    # test the full model pass
    if njmax > 0:
        model = SpookyModelVectorized(orig_atom_fea_len = orig_atom_fea_len,
                            atom_fea_len = atom_fea_len,
                            h_fea_len = 16,
                            n_h = 2,
                            global_fea_len = 2,
                            njmax = njmax)
    else:
        model = SpookyModel(orig_atom_fea_len = orig_atom_fea_len,
                            atom_fea_len = atom_fea_len,
                            h_fea_len = 16,
                            n_h = 2,
                            global_fea_len = 2)
    model.eval()

    t3 = time.time()
    out_model = model.forward(batch_orig_atom_fea, batch_nbr_fea_idx, batch_crystal_atom_idx,
                             batch_gs_fea, batch_gp_fea, batch_gd_fea, batch_global_fea)
    t3e = time.time()
    print('Model time: ', t3e-t3)

    #print(model.convs[0].spookylocal.Gs)

    return batch_gs_fea, batch_gp_fea, batch_gd_fea, out_model

    


if __name__ == "__main__":

    ########################################################################
    # model hyperparameters
    ########################################################################
    cutoff=5.5
    atom_fea_len = 8
    K=4
    num_crystal = 2
    njmax=0
    orig_atom_fea_len = 92
    seed = 0


    ########################################################################
    # Some crystal structures, uc and supercell representation 
    ########################################################################
    ciffile = sys.argv[1]
    structure = read(ciffile)
    crystal = Structure(structure.get_cell(),
                        structure.get_chemical_symbols(),
                        structure.get_positions(),
                        coords_are_cartesian=True) #Structure.from_file(ciffile)
    supercrystal = crystal *(5,5,5)


    # a dummy cubic crystal for testing rotations around origin
    cell = np.eye(3)*10
    symbs = ['C', 'O', 'N', 'S', 'H']
    positions1 = np.array(\
        [[ 0.0,  0.0,  0.0],  
        [ 1.0,  0.5,  1.7],
        [-2.1, -0.7,  1.2],
        [ 0.9, -1.4, -0.9],
        [-0.8,  1.2, -1.2]])
    #positions2
    #rotcrystal1 

    ########################################################################
    # Confirm rotation invariance of features
    ########################################################################
    # TODO


    ########################################################################
    # Confirm unit cell vs. supercell produces same prediction 
    ########################################################################
    print('\nDebug single supercell featurization:')
    gs_superuc, gp_superuc, gd_superuc, out_conv_superuc, out_model_superuc =\
        debug_featurization_single(supercrystal, cutoff, atom_fea_len, 
                                   K, njmax, orig_atom_fea_len, seed)

    print('\nDebug single uc featurization:')
    gs_uc, gp_uc, gd_uc, out_conv_uc, out_model_uc =\
        debug_featurization_single(crystal, cutoff, atom_fea_len, 
                                   K, njmax, orig_atom_fea_len, seed)
    
    
    # This actually won't help b/c the nbr order becomes different
    # based on the size of the uc representation
    #assert torch.allclose(gs_uc[0], gs_superuc[0])

    # Pymatgen supercells by [atom0]*num_reps + [atom1]*num_reps + ...
    #print([crystal[i].specie.number for i in range(len(crystal))])
    #print([supercrystal[i].specie.number for i in range(len(supercrystal))])

    # e.g. for any crystal with at least 2 atoms
    assert torch.allclose(out_conv_uc[0],out_conv_superuc[0])
    assert torch.allclose(out_conv_uc[-1],out_conv_superuc[-1])

    # e.g. atom feature vector can't depend on supercell size
    assert torch.allclose(out_model_uc[0], out_model_superuc[0])
    print('\nCompare defect model output of single vs supercell featurization:')
    print(out_model_uc, out_model_superuc)


    print('\nDebug batch uc featurization:')
    crystal_batch = [crystal, supercrystal]*2
    _, _, _, out_model = debug_featurization_batch(crystal_batch, cutoff, atom_fea_len,
                                                   K, njmax, orig_atom_fea_len, seed)
    print(out_model)


    njmax = 50
    print('\nDebug single supercell VECTORIZED featurization (njmax=%d):'%njmax)
    gs_superuc, gp_superuc, gd_superuc, out_conv_superuc, out_model_superuc =\
        debug_featurization_single(supercrystal, cutoff, atom_fea_len, 
                                   K, njmax, orig_atom_fea_len, seed)
    print(out_model_uc, out_model_superuc)

    print('\nDebug single uc VECTORIZED featurization (njmax=%d):'%njmax)
    gs_uc, gp_uc, gd_uc, out_conv_uc, out_model_uc =\
        debug_featurization_single(crystal, cutoff, atom_fea_len, 
                                   K, njmax, orig_atom_fea_len, seed)


    print('\nDebug batch uc VECTORIZED featurization (njmax=%d):'%njmax)
    _, _, _, out_model = debug_featurization_batch(crystal_batch, cutoff, atom_fea_len,
                                                   K, njmax, orig_atom_fea_len, seed)
    print(out_model)


