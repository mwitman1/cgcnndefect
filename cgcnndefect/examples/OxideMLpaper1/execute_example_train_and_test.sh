#! /bin/bash

# This example script shows how to use the defect CGCNN code, modified to directly 
# predict relaxed defect formation enthalpies from a host crystal structure.
# For details on methodology, please see the preprint: 
# Witman, Goyal, Ogitsu, McDaniel, Lany. ChemRXiv, 10.26434/chemrxiv-2022-frcns.

# For examples on how to:
# -automate bash scripting to run all cross-validations, encoding types, etc.
# -post-process all validation data
# -use additional DFT validation data on known water-splitting compounds
# -use the models to screen the Materials Project
# -post-process all screening data
# as demonstrated in the preprint, please see the Zenodo repository:
# Witman, Goyal, Ogitsu, McDaniel, Lany. Zenodo, 10.5281/zenodo.5999073

# This example can be adapted via the following which are
# used to validate the model, test different encoding strategies, etc.
# 1. fraction of data made available [0.10, 0.40, or 1.00]
# 2. choice of how to initially encode node features based only on element id  [atom_init.json.*]
# 3. choice of adding (computed, site-specific) oxidation states to node features (*.locals)
# 4. choice of adding computed host properties to defect node feature vector (*.globals)

# Training hyperparameters that are constant across all models
# production models should be trained for 1000 epochs with the best model determined by early stopping
hyperparam_flags="--h-fea-len 16 --atom-fea-len 8 --n-conv 2 --epochs 1000 --lr 0.05 --optim Adam --n-h 2 --disable-cuda --test-ratio 0.02 --seed 1000"

function makedir {
    if [ ! -d ./$1 ]
    then
        mkdir ./$1
    fi
}

cd cgcnn

    ######################################################################################
    # 1. This trains a model for:
    ######################################################################################

    # atom_init.json contains the original onehot encoding of elemental properties for initial node features
    # *.locals contains the onehot encoding of specific atom site features, e.g. oxidation states, for each structure
    # *.globals contains the host compound features (e.g. dH_f, e- effective mass, band gap) for each structure
    encode_type="_ve_vg_vs" # (elemental, global, and oxidation state features)
    encode_flags="--init-embed-file atom_init.json --atom-spec locals --crys-spec globals" 

    # Using all available data and train the k=0 model of the defectwise, K=10 CV scheme
    train_struct_flag="--csv-ext .train_1.00k0"

    # Save the model to the corresponding location
    resultdir="model-1.00k0""$encode_type" && makedir $resultdir
    resultdir_flag="--resultdir ""$resultdir"

    printf '\n*************\nTraining Model 1\n*************\n\n'
    cgcnn-defect-train . $hyperparam_flags $encode_flags $train_struct_flag $resultdir_flag


    ######################################################################################
    # 2. Then validates on the corresponding hold out set:
    ######################################################################################

    hold_struct_flag="--csv-ext .hold_1.00k0"
    model_loc_flag="$resultdir"/model_best.pth.tar
    data_loc_flag="."
    CIFfeaturizer_loc_flag="-CIFdatapath "$resultdir"/dataset.pth.tar"

    printf '\n*************\nValidating Model 1\n*************"\n\n'
    cgcnn-defect-predict $model_loc_flag $data_loc_flag $CIFfeaturizer_loc_flag $hold_struct_flag $resultdir_flag --disable-cuda

    ######################################################################################
    # 3. This trains a model similar to the above, with the exception that the train/test
    # stratification has been perfomed structurewise/compoundwise (not defectwise)
    ######################################################################################

    encode_type="_struct_ve_vg_vs"
    encode_flags="--init-embed-file atom_init.json --atom-spec locals --crys-spec globals" 
    resultdir="model-1.00k0""$encode_type" && makedir $resultdir
    resultdir_flag="--resultdir ""$resultdir"
    train_struct_flag="--csv-ext .train_1.00k0_struct"

    printf '\n*************\nTraining Model 2\n*************\n\n'
    cgcnn-defect-train . $hyperparam_flags $encode_flags $train_struct_flag $resultdir_flag


    ######################################################################################
    # 4. Then validates on the corresponding hold out set:
    ######################################################################################

    hold_struct_flag="--csv-ext .hold_1.00k0_struct"
    model_loc_flag="$resultdir"/model_best.pth.tar
    data_loc_flag="."
    CIFfeaturizer_loc_flag="-CIFdatapath "$resultdir"/dataset.pth.tar"

    printf '\n*************\nValidating Model 2\n*************\n\n'
    cgcnn-defect-predict $model_loc_flag $data_loc_flag $CIFfeaturizer_loc_flag $hold_struct_flag $resultdir_flag --disable-cuda


    ######################################################################################
    # 3. Compare the holdout test MAE between the two models
    ######################################################################################

    


cd ..
