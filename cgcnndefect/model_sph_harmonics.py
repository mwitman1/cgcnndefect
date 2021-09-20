#! /usr/bin/env python3

from pymatgen.core.structure import Structure
from scipy.special import sph_harm
from math import comb
from pprint import pprint
import numpy as np
import sys
import torch
import torch.nn as nn
from typing import Tuple, List
import time


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

def compute_rhok(r, k, K, rcut, gamma=0.5):

    rhok = comb(K-1,k)*np.exp(-gamma*r)**k*(1-np.exp(-gamma*r))**(K-1-k)*\
           fcut(r,rcut) 

    return rhok
           

def get_harmonics_fea(structure, all_nbrs, K, rcut, njmax=75):
    """
    May need to optimize/vectorize as much of this as possible later on

    Nc - number of atoms in crystal
    Nj - number of neighbors of atom i

    returns 
        gs_fea : list of size N of np.arrays with shape(
    """

    gs_fea, gp_fea, gd_fea = [], [], []
    nbr_dist, nbr_fea_idx, nbr_vec = [], [], []
    nbr_angles = []
    for i, nbrs in enumerate(all_nbrs):
        # Using sph_harm basis set, no longer need to artifically cap the 
        # number of interactions (assigned bonds) to a fixed max value
        # within the cutoff, so no need to sort
        # However, zero-padding up to max neighbors allows vectorization
        # of all operations despite the non-constant num of neighbors between atoms
        # Therefore, a present njmax is being used, and program will automatically
        # exit if more neighbors exist in the cutoff than 
        if len(nbrs) >= njmax:
            raise ValueError("Error! You're chosen cutoff contains more than max num of possible neighbors")

        # variable number of neighbors w/in site for each site
        nbr_fea_idx.append([nbr[2] for nbr in nbrs])
         
        # Need to be especially careful to use Pymatgen Periodic neighbor
        # to get correct distance vector
        icoords = structure[i].coords
        nbr_vec = np.array([nbr[0].coords-icoords for nbr in nbrs])
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
        nbr_gs_fea_pad=np.zeros((njmax-len(nbrs),K,1))

        # nbr_gp_fea with shape(nj, K, 3)
        nbr_gp_fea=np.stack(((nbr_gp_1_n1*nbr_rhok).T,
                             (nbr_gp_1_0*nbr_rhok).T,
                             (nbr_gp_1_1*nbr_rhok).T),axis=2)
        nbr_gp_fea_pad=np.zeros((njmax-len(nbrs),K,3))

        
        # nbr_gp_fea with shape(nj, K, 5)
        nbr_gd_fea=np.stack(((nbr_gd_2_n2*nbr_rhok).T,
                             (nbr_gd_2_n1*nbr_rhok).T,
                             (nbr_gd_2_0*nbr_rhok).T,
                             (nbr_gd_2_1*nbr_rhok).T,
                             (nbr_gd_2_2*nbr_rhok).T),axis=2)
        nbr_gd_fea_pad=np.zeros((njmax-len(nbrs),K,5))

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

        #gs_fea.append(torch.Tensor(nbr_gs_fea).unsqueeze(-1))
        #gp_fea.append(torch.Tensor(nbr_gp_fea))
        #gd_fea.append(torch.Tensor(nbr_gd_fea))

        # if padding 0's up to njmax
        gs_fea.append(torch.Tensor(np.concatenate((nbr_gs_fea,nbr_gs_fea_pad),axis=0)))
        gp_fea.append(torch.Tensor(np.concatenate((nbr_gp_fea,nbr_gp_fea_pad),axis=0)))
        gd_fea.append(torch.Tensor(np.concatenate((nbr_gd_fea,nbr_gd_fea_pad),axis=0)))


    return gs_fea, gp_fea, gd_fea

class Residual(nn.Module):
    def __init__(self,fea_len_in, fea_len_out):
        super(Residual, self).__init__()
        self.linear1 = nn.Linear(fea_len_in,fea_len_out)
        self.silu1 = nn.SiLU()

    def forward(self,x):
        return x + self.linear1(self.silu1(x))

class ResMLP(nn.Module):
    def __init__(self,fea_len_in, fea_len_out):
        super(ResMLP, self).__init__()
        self.linear1 = nn.Linear(fea_len_in,fea_len_out)
        self.silu1 = nn.SiLU()
        self.residual = Residual(fea_len_in,fea_len_in)

    def forward(self,x):
        return self.linear1(self.silu1(self.residual(x)))

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
                      crystal_atom_idx : List[torch.LongTensor],
                      gs : List[torch.Tensor],
                      gp : List[torch.Tensor],
                      gd : List[torch.Tensor]):
        """
        Test forward vectorized using 0 padding for a pre-selected nj max
        """
        # TODO
        pass

    def forward(self, atom_fea : torch.Tensor,
                      nbr_fea_idx : List[torch.LongTensor],
                      crystal_atom_idx : List[torch.LongTensor],
                      gs : List[torch.Tensor],
                      gp : List[torch.Tensor],
                      gd : List[torch.Tensor]):

        """
        N : number of atoms in batch
        N0 : number of crystals in batch
        nc : number of atoms in crystal c in crystal_atom_idx
        nj : number of neighbors to atom i in batch

        atom_fea : Variable(torch.Tensor) shape (N, atom_fea_len)
        nbr_fea_idx : List(torch.LongTensor) of len(N) with shape(nj)
        crystal_atom_idx : List(torch.LongTensor) of len(N) with shape(nc)

        gs : List(torch.Tensor) of len (N) with shape(nj, K)
        gp : List(torch.Tensor) of len (N) with shape(nj, K, 3)
        gd : List(torch.Tensor) of len (N) with shape(nj, K, 5)

        """
        # TODO : need conversion from crystal_atom_idx to nbr_fea_idx
        # for when there are multiple different crystals in the batch

        #nbr_fea_idx_in_batch = ...

        # shape (N, atom_fea_len)
        s_fea = self.resmlp_s(atom_fea)
        # shape (N, atom_fea_len)
        p_fea = self.resmlp_p(atom_fea)
        # shape (N, atom_fea_len)
        d_fea = self.resmlp_d(atom_fea)
       
        ## OPTION 1: vectorized version
        ## -> when each atom env has padded 0s to make up to exactly njmax neighbors
 
        ##print(self.Gs.expand(len(gs),gs[0].shape[0],self.Gs.shape[0],self.Gs.shape[1]).shape)
        ##print(torch.stack(gs,dim=0).shape, len(gs))
        ## shape (N, njmax, F, 1)
        #s_env = torch.matmul(self.Gs.expand(len(gs), # N
        #                                    gs[0].shape[0], # njmax
        #                                    self.Gs.shape[0], # F
        #                                    self.Gs.shape[1]), # K 
        #                     torch.stack(gs,dim=0)) # (N, njmax, K, 1)
        ##print(s_env.shape)
        ## shape (N, njmax, F) # TODO: how the heck to vectorize
        #nbr_s_fea = torch.stack([s_fea.index_select(0,nbr_fea_idx[i])\
        #                         for i in range(len(nbr_fea_idx))])
        ##print(nbr_s_fea.shape)
        ## shape (N,atom_fea)
        #final_s = torch.sum(nbr_s_fea.unsqueeze(-1)*s_env,dim=1).squeeze()
        ##print(final_s_vectorized.shape)
        #

        ## shape (N, nj, F, 3)
        #p_env = torch.matmul(self.Gp.expand(len(gs), # N
        #                                    gp[0].shape[0], # njmax
        #                                    self.Gp.shape[0], # F
        #                                    self.Gp.shape[1]), # K
        #                     torch.stack(gp,dim=0)) # (N, njmax, K, 3)
        ##print(p_env.shape)
        ## shape (N, nj, F) # TODO how the heck to vectorize
        #nbr_p_fea = torch.stack([p_fea.index_select(0, nbr_fea_idx[i])\
        #                        for i in range(len(nbr_fea_idx))])
        ##print(nbr_p_fea.shape)
        ## shape (N, F, 3)  
        #all_p_vectorized = torch.sum(nbr_p_fea.unsqueeze(-1).expand(\
        #             p_env.shape[0], p_env.shape[1], p_env.shape[2], p_env.shape[3])*p_env,dim=1)
        ##print(all_p_vectorized.shape)
        #all_p = torch.transpose(all_p_vectorized,1,2)


        ## shape (N, nj, F, 5)
        #d_env = torch.matmul(self.Gd.expand(len(gd), # N
        #                                    gd[0].shape[0], # njmax
        #                                    self.Gd.shape[0], # F
        #                                    self.Gd.shape[1]), # K
        #                     torch.stack(gd,dim=0)) # (N, njmax, K, 5)
        ## shape (N, nj, F)
        #nbr_d_fea = torch.stack([d_fea.index_select(0, nbr_fea_idx[i])\
        #                        for i in range(len(nbr_fea_idx))])
        ## shape (N, 5) 
        #all_d_vectorized = torch.sum(nbr_d_fea.unsqueeze(-1).expand(\
        #             d_env.shape[0], d_env.shape[1], d_env.shape[2], d_env.shape[3])*d_env,dim=1)
        #all_d = torch.transpose(all_d_vectorized,1,2)


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

        # shape(N, atom_fea_len)
        final_c = self.resmlp_c(atom_fea)

        #print(final_c)
        #print(final_s)
        #print(final_p)
        #print(final_d)

        # TODO: BOTH APPROACHES MUST GIVE SAME RESULT !!!!!!
        #print(final_s - final_s_vectorized)
        #print('Max diff s: ', torch.max(final_s - final_s_vectorized))
        #assert torch.allclose(final_s, final_s_vectorized, atol=1e-4)
        #print(torch.transpose(all_p,1,2) - all_p_vectorized)
        #print('Max diff p: ', torch.max(torch.transpose(all_p,1,2) - all_p_vectorized))
        #assert torch.allclose(torch.transpose(all_p,1,2), all_p_vectorized, atol=1e-4)
        ##assert torch.allclose(all_p, all_p_vectorized)



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
        

    def forward(self, atom_fea, nbr_fea_idx, crystal_atom_idx,
                      gs_fea, gp_fea, gd_fea):

        intermed_atom_fea = self.res1(atom_fea)
        local = self.spookylocal(intermed_atom_fea, nbr_fea_idx,
                                 crystal_atom_idx, gs_fea, gp_fea, gd_fea)
        atom_fea = self.res2(intermed_atom_fea + local)
        
        return atom_fea


class SpookyModel(nn.Module):

    def __init__(self, orig_atom_fea_len, atom_fea_len=64, n_conv=3, 
                       h_fea_len=128, n_h=1, K=4, all_elems = [0], global_fea_len=0):
        super(SpookyModel, self).__init__()

        self.n_h = n_h

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
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
            atom_fea = conv_func(atom_fea, nbr_fea_idx, crystal_atom_idx,
                                 gs_fea, gp_fea, gd_fea)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        crys_fea = self.conv_to_fc(torch.cat([crys_fea, global_fea],dim=1))

        if self.n_h > 1:
            for fc in self.fcs:
                crys_fea = fc(crys_fea)

        out = self.fc_out(crys_fea)

        return [out]

    def pooling(self, atom_fea : torch.Tensor,
                      crystal_atom_idx : List[torch.Tensor]):

        # TODO implement flexibility for pooling type from function arguments

        # If pooling requires a summing/avging/etc over all atoms in crystal
        #summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
        #              for idx_map in crystal_atom_idx]

        # If, instead of pooling, simply want the feature vec of a single atom
        summed_fea = [torch.index_select(atom_fea,0,idx_map[0])\
                      for idx_map in crystal_atom_idx]

        # If we want to pool the feature vecs of only select atoms
        # etc

        return torch.cat(summed_fea,dim=0)

if __name__ == "__main__":

    ########################################################################
    # model hyperparameters
    ########################################################################
    cutoff=5.5
    atom_fea_len = 8
    K=4
    ciffile = sys.argv[1]
    num_crystal = 2
    njmax=50
    torch.manual_seed(0)

    ########################################################################
    # example for one crystal
    ########################################################################
    t0 = time.time()
    crystal = Structure.from_file(ciffile)
    crystal_batch = [crystal for _ in range(num_crystal)]
    #all_atom_types = [crystal[i].specie.symbol for i in range(len(crystal))]
    all_nbrs = crystal.get_all_neighbors(cutoff, include_index=True)

    # network inputs
    dummy_init_atom_fea = torch.Tensor([[i for _ in range(atom_fea_len)]\
                                 for i in range(len(crystal))])
    nbr_fea_idx = [torch.LongTensor(list(map(lambda x: x[2],nbr))+\
                   [0] * (njmax - len(nbr)))\
                   for nbr in all_nbrs]

    crystal_atom_idx = [torch.LongTensor(np.arange(len(crystal)))]
    t0e = time.time()

    # time the featurization
    t1 = time.time()
    gs_fea, gp_fea, gd_fea = get_harmonics_fea(crystal, all_nbrs, 
                                               K, cutoff)
    t1e = time.time()

    # time a single convolution operation
    l1 = SpookyLocalBlock(atom_fea_len,K)
    out = l1.forward(dummy_init_atom_fea, nbr_fea_idx, crystal_atom_idx, 
                     gs_fea, gp_fea, gd_fea)

    # time entire model evaluation (for 1 crystal)
    model = SpookyModel(orig_atom_fea_len = atom_fea_len,
                        atom_fea_len = atom_fea_len,
                        h_fea_len = 16,
                        n_h=2,
                        global_fea_len=2)

    model.eval()
    global_fea = torch.Tensor([[1,2]])
    t2 = time.time()
    out = model.forward(dummy_init_atom_fea, nbr_fea_idx, crystal_atom_idx, 
                        gs_fea, gp_fea, gd_fea, global_fea)
    t2e = time.time()

    print(out)
    print('Crystal processing time: ', t1e-t1)
    print('Featurization time: ', t1e-t1)
    print('Model time: ', t2e-t2)


    ########################################################################
    # example for one crystal
    ########################################################################

    # example for batch of crytals
    batch_atom_fea, batch_nbr_fea_idx, batch_crystal_atom_idx = [], [], []
    batch_gs_fea, batch_gp_fea, batch_gd_fea  = [], [], []
    batch_global_fea = []

    batch_size = len(crystal_batch)
    base_idx = 0
    for i,crystal in enumerate(crystal_batch):
        n_i = len(crystal_batch[i])
        batch_crystal_atom_idx.append(torch.LongTensor(np.arange(n_i)+base_idx))
        
        batch_atom_fea.append(torch.Tensor([[i for _ in range(atom_fea_len)]\
                                            for i in range(len(crystal))]))
        batch_nbr_fea_idx+=[base_idx + torch.LongTensor(list(map(lambda x: x[2],nbr))+\
                                                        [0]*(njmax-len(nbr)))\
                            for nbr in all_nbrs]
        
        gs_fea, gp_fea, gd_fea = get_harmonics_fea(crystal, all_nbrs, 
                                                   K, cutoff)

        batch_gs_fea+=gs_fea
        batch_gp_fea+=gp_fea
        batch_gd_fea+=gd_fea

        base_idx += n_i

    batch_atom_fea = torch.cat(batch_atom_fea, dim=0)
    batch_global_fea = torch.stack([torch.Tensor([1,2]) for _ in range(batch_size)], dim=0)
    target = torch.stack([torch.Tensor([1]) for _ in range(batch_size)], dim=0)

    t3 = time.time()
    out = model.forward(batch_atom_fea, batch_nbr_fea_idx, batch_crystal_atom_idx,
                        batch_gs_fea, batch_gp_fea, batch_gd_fea, batch_global_fea)
    t3e = time.time()
    print('Model forward (2 crystals): ', t3e-t3)

    #optimizer = torch.optim.Adam(model.parameters(), 0.05)
    #optimizer.zero_grad()
    #criterion = nn.MSELoss()

    #t4 = time.time()
    #loss = criterion(out[0],target)
    #loss.backward()
    #optimizer.step()
    #t4e = time.time()
    #print('Model backward (2 crystals): ', t4e-t4)

    #num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("# trainable params: %d"%num_params)

    #print(sys.getsizeof(batch_gs_fea),
    #      sys.getsizeof(batch_gp_fea),
    #      sys.getsizeof(batch_gd_fea),
    #      sys.getsizeof(batch_atom_fea))


    #print("debugging neighbor padding")
    #testF = 5
    #testnj = 7
    #testnjmax = 10
    #testK = 4
    #np.random.seed(0)
    #testgp = [i+torch.tensor(np.random.rand(testK, 3))\
    #          for i in range(testnj)]
    #testgp_padded = testgp + [torch.tensor(np.zeros((testK, 3)))\
    #                          for i in range(testnjmax - testnj)]
    #testgp = torch.stack(testgp)
    #testgp_padded = torch.stack(testgp_padded)
    #testGp = torch.tensor(np.random.rand(testF, testK))

    #res = torch.matmul(testGp, testgp[0])

    #print(testGp, testgp)
    #print(testGp.shape, testgp.shape)
    #print(res)

    

    






















