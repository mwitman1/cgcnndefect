import os
import argparse
import time

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ase.build import bulk
from ase.io import read,write

from .model import CrystalGraphConvNet
from .data import CIFData
from .command_line_predict import Normalizer


class CGCNN_IAP(object):

    def __init__(self, modelpath, datapath, cuda=False):

        # Dummy data to initialize the CIFData object
        # TODO:
        # eventually will need a big refactor
        # a cached CIFData object (for featurizing crystals) 
        # needs to be loaded alongside the trained model, 
        # as well as the atom_init.json file used during training
        #dummy = bulk('Cu', 'fcc', a=3.6)
        #write(dummy, os.path.join(datapath,"dummy.cif"))
        #with open(os.path.join(datapath,'id_prop.csv'),'w') as f:
        #    f.write("dummy 0")
        self.dataset = CIFData(datapath)
        structures, _, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        
        self.model, self.model_args, self.normalizer = \
            self._load_model_normalizer(modelpath, orig_atom_fea_len, nbr_fea_len)
    
    def _load_model_normalizer(self, modelpath, orig_atom_fea_len, nbr_fea_len):

        # Load requested model into pytorch and get its parameters
        if os.path.isfile(modelpath):
            print("=> loading model params '{}'".format(modelpath))
            model_checkpoint = torch.load(modelpath,
                                          map_location=lambda storage, 
                                          loc: storage)
            model_args = argparse.Namespace(**model_checkpoint['args'])
            print("=> loaded model params '{}'".format(modelpath))
        else:
            print("=> no model params found at '{}'".format(modelpath))
            raise Exception("Please provide valid model file")

        # no point in having a classification task for an IAP
        if model_args.task == 'classification':
            raise Exception("Interatomic potential can't be based off of a "
                            "classification model")

        # initialize model
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=False,
                                    Fxyz=True if model_args.task == 'Fxyz'\
                                              else False)

        # Load a checkpointed model
        if os.path.isfile(modelpath):
            print("=> loading model '{}'".format(modelpath))

            checkpoint = torch.load(modelpath,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer = Normalizer(torch.zeros(3))
            # TODO normlazer_Fxyz = ...
            normalizer.load_state_dict(checkpoint['normalizer'])

        return model, model_args, normalizer 

    def predict(self, aseatoms):

        start = time.time()
        structure = AseAtomsAdaptor.get_structure(aseatoms)
        end = time.time()
        print('Structure convert (s): %.6f'%(end-start))

        # featurize this new structure
        start = time.time()
        input, _, _, _ = self.dataset.featurize_crystal(structure)

        # crystal_atom_idx (as in collate_pool) for only one crystal structure
        last_inp = [torch.LongTensor(np.arange(len(structure)))]
        end = time.time()
        print('Featurization (s): %.6f'%(end-start))

        start = time.time()
        with torch.no_grad():
            if self.model_args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in\
                                                                   last_inp])
            else: 
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             last_inp)

            # model prediction
            output = self.model(*input_var)
        end = time.time()
        print('Energy call (s): %.6f'%(end-start))

        return output[0]
        
