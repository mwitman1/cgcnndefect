import torch
import torch.nn as nn

N=3 # 3 atoms
atom_fea_len = 2 # size of atom vector
M=4 # Max number of neighbors
nbr_fea_len = 1 # size of edge vector 

# N=3 atoms with atom_fea_len=2
atom_in_fea = torch.Tensor([[0.,0.],[0.1,0.1],[0.2,0.2]])
print('\natom_in_fea = ', atom_in_fea.shape)
print(atom_in_fea)

# N=3 atoms with M=4 neighbors with nbr_fea_len=1
nbr_fea = torch.Tensor([ # atom 1
                        [ # nbr 1
                         [1.11],
                         # nbr2
                         [1.21],
                         # nbr 3
                         [1.11],
                         # nbr 4
                         [1.21]
                        ],
                        # atom 2
                        [ 
                         [1.01],[1.21],[1.01],[1.21]
                        ],
                        # atom 3
                        [
                         [1.11],[1.21],[1.11],[1.21]
                        ]
                       ])
print('\nnbr_fea = ', nbr_fea.shape)
print(nbr_fea)


# N=3 atoms with indices of neighbors
nbr_fea_idx = torch.LongTensor([[1,2,1,2],
                                [0,2,0,2],
                                [0,1,0,1]])


# N=3 atoms with M=4 neighbors with atom_in_fea=2 
atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
print('\natom_nbr_fea = ', atom_nbr_fea.shape)
print(atom_nbr_fea)

# N=3 atoms with M=4 neighbors with 5 features: atom_in_fea[i] (2), atom_in_fea[j] (2), b_ij (1)
total_nbr_fea = torch.cat(
    [atom_in_fea.unsqueeze(1).expand(N, M, atom_fea_len),
     atom_nbr_fea, nbr_fea], dim=2)
print('\ntotal_nbr_fea = ', total_nbr_fea.shape)
print(total_nbr_fea)

# N=3 atoms with M=4 neighbors with 2*(atom_fea_len=2)=4 features: fc_full output 
fc_full = nn.Linear(2*atom_fea_len+nbr_fea_len,2*atom_fea_len)
total_gated_fea = fc_full(total_nbr_fea)
print('\ntotal_gated_fea = ', total_gated_fea.shape)

bn1 = nn.BatchNorm1d(2*atom_fea_len)
total_gated_fea = bn1(total_gated_fea.view(
    -1, atom_fea_len*2)).view(N, M, atom_fea_len*2)
print('bn1 = ', total_gated_fea.view(-1,atom_fea_len*2).shape)
print('\ntotal_gated_fea = ', total_gated_fea.shape)

nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
print('\nnbr_filter = ', nbr_filter.shape)
print('\nnbr_core = ', nbr_core.shape)

bn2 = nn.BatchNorm1d(atom_fea_len)

