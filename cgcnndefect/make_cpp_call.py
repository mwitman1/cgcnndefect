#! /usr/bin/env python3

import sys
import pickle

def get_features(fname, all_atom_types, all_nbrs):
    """
    all_atom_types : list of ints
        list of atomic numbers for each bead
    all_nbrs : list of list of (nbr atomic num, nbr dist, nbr site index)
        Can just obtained by small restructuring of FFEAST neighbor list
    """
    with open(fname, 'rb') as f:
        dataset = pickle.load(f)

    feat_tup = dataset.featurize_from_nbr_and_atom_list(all_atom_types,
                                                        all_nbrs)
    return feat_tup


if __name__ == '__main__':
    test_all_atom_types = [1, 12]
    test_all_nbrs = [
                      [
                        [12, 0.1, 1]
                      ],
                      [
                        [1,0.1,0]
                      ]
                    ]
    fea0, fea1, fea2, fea3, fea4, fea5, fea6, fea7 =\
        get_features(sys.argv[1],test_all_atom_types,test_all_nbrs)

    print(tup)


# I think this part needs to be done entirely in C++ via torch script
#with open(sys.argv[2],'rb') as f:
#    checkpoint = torch.load(f,map_location=lambda storage,loc: storage)
#    model.load_state_dict(checkpoint['state_dict'])
#    normalizer.load_state_dict(checkpoint['normalizer'])
#
#output = model(*feat_tup)
#ener = normalizer.denorm(output[0].data.cpu())
#
#print(ener)
