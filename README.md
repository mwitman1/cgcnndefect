# CGCNN for directly predicting relaxed vacancy formation enthalpies

Modifications for predicting defects and other various modifications have been made to make some tasks easier: cross-validation, running on cluster, etc. 

## System requirements:
Python3, pip3 (python package manager), and command line execution of python programs on MacOS, Linux, or Windows

## Tested architecture and package versions:
```
Last updated: 2022-09-29T15:25:05.106502-07:00

Python implementation: CPython
Python version       : 3.9.7
IPython version      : 7.24.1

Compiler    : Clang 11.0.0 (clang-1100.0.33.17)
OS          : Darwin
Release     : 20.6.0
Machine     : x86_64
Processor   : i386
CPU cores   : 12
Architecture: 64bit
pymatgen : 2022.9.21
ase      : 3.22.0
torch    : 1.9.0
numpy    : 1.23.1
sklearn  : 0.0 
watermark: 2.3.1
Name: scikit-learn
Version: 0.24.2
```


## Installation guide
First clone the repository 
```
git clone --single-branch --branch Paper1 https://github.com/mwitman1/cgcnndefect
```
To install the package, please use:
```bash
cd cgcnndefect
pip install -e .
```
which allows you to then execute training or prediction tasks from anywhere using the command line:
```bash
cgcnn-defect-train $flags
cgcnn-defect-predict $flags
```
Install time is a few seconds.

## Demo
The training data files from the full data repository (https://zenodo.org/record/5999073) has been provided in the examples directory:
```
cgcnndefect/examples/OxideMLpaper1
```
Specifically, all input files needed to run the defect CGCNN code have already been prepared in: 
```
cgcnndefect/examples/OxideMLpaper1/cgcnn
```
These have been prepared from output DFT data provided in (*/mags, */poscars, */csvs, */oxstate).

The ```cgcnn``` directory contains a python script to check your watermark and a bash script to execute a demo of the training and validation procedure  in ```execute_example_train_and_test.sh```:


```bash
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
    # Note this command writes test results to test_results.csv
    cgcnn-defect-train . $hyperparam_flags $encode_flags $train_struct_flag $resultdir_flag


    ######################################################################################
    # 2. Then validates on the corresponding hold out set:
    ######################################################################################

    hold_struct_flag="--csv-ext .hold_1.00k0"
    model_loc_flag="$resultdir"/model_best.pth.tar
    data_loc_flag="."
    CIFfeaturizer_loc_flag="-CIFdatapath "$resultdir"/dataset.pth.tar"

    printf '\n*************\nValidating Model 1\n*************"\n\n'
    # Note this command writes test results to all_results.csv
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
    # Note this command writes test results to test_results.csv
    cgcnn-defect-train . $hyperparam_flags $encode_flags $train_struct_flag $resultdir_flag


    ######################################################################################
    # 4. Then validates on the corresponding hold out set:
    ######################################################################################

    hold_struct_flag="--csv-ext .hold_1.00k0_struct"
    model_loc_flag="$resultdir"/model_best.pth.tar
    data_loc_flag="."
    CIFfeaturizer_loc_flag="-CIFdatapath "$resultdir"/dataset.pth.tar"

    printf '\n*************\nValidating Model 2\n*************\n\n'
    # Note this command writes test results to all_results.csv
    cgcnn-defect-predict $model_loc_flag $data_loc_flag $CIFfeaturizer_loc_flag $hold_struct_flag $resultdir_flag --disable-cuda
    
cd ..
```

Executed on a typical desktop this training may take up to around an hour. The progress of the training will be output to the console, and you can expect to see
```
*************
Validating Model 1
*************

=> loading model 'model-1.00k0_ve_vg_vs/model_best.pth.tar'
=> loaded model 'model-1.00k0_ve_vg_vs/model_best.pth.tar' (epoch 763, validation 0.4224497079849243)
Test: [0/1]	Time 2.282 (2.282)	Loss 0.0869 (0.0869)	MAE 0.547 (0.547)
 ** MAE 0.547
```

and

```
*************
Validating Model 2
*************

=> loading model 'model-1.00k0_struct_ve_vg_vs/model_best.pth.tar'
=> loaded model 'model-1.00k0_struct_ve_vg_vs/model_best.pth.tar' (epoch 108, validation 0.4658649265766144)
Test: [0/1]	Time 3.051 (3.051)	Loss 0.0523 (0.0523)	MAE 0.525 (0.525)
 ** MAE 0.525
```

Note these test results are inclusive of O and non-O defects (as evident from the structure names in the */all_results.csv file).


## Advanced use instructions

Most of the advanced instructions for CGCNN use can be found in the original README (see below). Here are a few details to assist with using the modifications in this repository to reproduce the results in (https://chemrxiv.org/engage/chemrxiv/article-details/628bdf9f87d01f60fcefa355). Note the large number of model trainings needed to cross-validate and test the different model types will require significant compute time and will be best performed on an HPC system or a powerful desktop with a large number of CPUs. All defect data, the training/validation bash scripts that execute this code, and jupyter notebooks for post-processing the training and materials screening data are provided in the Zenodo repository (https://zenodo.org/record/5999073).   

### Managing cross-validation
The data has already been split for the various K-fold cross validations in ```id_prop.csv.*```. Structures supplied during training (```cgcnn-defect-train $flags```) or inference (```cgcnn-defect-predict $flags```) can be controlled via the flag:
```bash
--csv-ext .your_csv_ext
```
Note (0.10, 0.4, 1.00) means using 10, 40, or 100% of the available data, k{0..9} is the particular fold, "train" is the split used for train/validation splits with early stopping, and "hold" is the hold out split used for final model accuracy evaluation. A "_struct" indicates that the splits were generated structurewise/compoundwise, i.e., defects from the same structure must be only in either the "train" or "hold" split. 

### Defect modifications
In this version, the pooling function has been *hard coded* to only extract the feature vector of the node at index i=0 (the atom to be defected). This will be updated in the development branch and in future releases to be more efficient and flexible.

### Initial feature encoding
One can investigate the performance of different initial feature encoding schemes: initial node features are based on elemental properties encoded in various ways (```atom_init.json.*```), addition of node-specific properties like oxidation states (data in ```*.locals```), or global crystal features (data in ```*.globals```)

- To change elemenent encoding file
```bash
--init-embed-file $your_atom_init.json
```
- For a given structure1.cif, can introduce local node attributes (e.g. oxidation state) contained in structure1.cif.locals at the graph encoding stage via:
```bash
--atom-spec locals
``` 
- For a given structure1.cif, can introduce global features (e.g. compound formation enthalpy) contained in structure1.cif.globals at the graph encoding stage via:
```bash
--crys-spec globals
``` 

### How to cite

Please cite the following work if you want to use CGCNN and defect modifications.

```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
@article{Witman2022,
  author = {Witman, Matthew D. and Goyal, Anuj and Ogitsu, Tadashi and McDaniel, Anthony H. and Lany, Stephan},
  doi = {10.26434/chemrxiv-2022-frcns},
  journal = {ChemRxiv},
  pages = {10.26434/chemrxiv-2022-frcns},
  title = {{Graph neural network modeling of vacancy formation enthalpy for materials discovery and its application in solar thermochemical water splitting}},
  year = {2022},
  url = {https://chemrxiv.org/engage/chemrxiv/article-details/628bdf9f87d01f60fcefa355}
}
```

# Crystal Graph Convolutional Neural Networks

This software package implements the Crystal Graph Convolutional Neural Networks (CGCNN) that takes an arbitary crystal structure to predict material properties. 

The package provides two major functions:

- Train a CGCNN model with a customized dataset.
- Predict material properties of new crystals with a pre-trained CGCNN model.

The following paper describes the details of the CGCNN framework:

[Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a CGCNN model](#train-a-cgcnn-model)
  - [Predict material properties with a pre-trained CGCNN model](#predict-material-properties-with-a-pre-trained-cgcnn-model)
- [Data](#data)
- [Authors](#authors)
- [License](#license)

## How to cite

Please cite the following work if you want to use CGCNN.

```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
```

##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `cgcnn` and install all prerequisites:

```bash
conda upgrade conda
conda create -n cgcnn python=3 scikit-learn pytorch torchvision pymatgen -c pytorch -c conda-forge
```

*Note: this code is tested for PyTorch v1.0.0+ and is not compatible with versions below v0.4.0 due to some breaking changes.

This creates a conda environment for running CGCNN. Before using CGCNN, activate the environment by:

```bash
source activate cgcnn
```

Then, in directory `cgcnn`, you can test if all the prerequisites are installed properly by running:

```bash
python main.py -h
python predict.py -h
```

This should display the help messages for `main.py` and `predict.py`. If you find no error messages, it means that the prerequisites are installed properly.

After you finished using CGCNN, exit the environment by:

```bash
source deactivate
```

## Usage

### Define a customized dataset 

To input crystal structures to CGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

There are two examples of customized datasets in the repository: `data/sample-regression` for regression and `data/sample-classification` for classification. 

**For advanced PyTorch users**

The above method of creating a customized dataset uses the `CIFData` class in `cgcnn.data`. If you want a more flexible way to input crystal structures, PyTorch has a great [Tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#sphx-glr-beginner-data-loading-tutorial-py) for writing your own dataset class.

### Train a CGCNN model

Before training a new CGCNN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

Then, in directory `cgcnn`, you can train a CGCNN model for your customized dataset by:

```bash
python main.py root_dir
```

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. For instance, `data/sample-regression` has 10 data points in total. You can train a model by:

```bash
python main.py --train-size 6 --val-size 2 --test-size 2 data/sample-regression
```
or alternatively
```bash
python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/sample-regression
```

You can also train a classification model with label `--task classification`. For instance, you can use `data/sample-classification` by:

```bash
python main.py --task classification --train-size 5 --val-size 2 --test-size 3 data/sample-classification
```

After training, you will get three files in `cgcnn` directory.

- `model_best.pth.tar`: stores the CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Predict material properties with a pre-trained CGCNN model

Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` for all the crystal structures that you want to predict.
- Obtain a [pre-trained CGCNN model](pre-trained) named `pre-trained.pth.tar`.

Then, in directory `cgcnn`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar root_dir
```

For instace, you can predict the formation energies of the crystals in `data/sample-regression`:

```bash
python predict.py pre-trained/formation-energy-per-atom.pth.tar data/sample-regression
```

And you can also predict if the crystals in `data/sample-classification` are metal (1) or semiconductors (0):

```bash
python predict.py pre-trained/semi-metal-classification.pth.tar data/sample-classification
```

Note that for classification, the predicted values in `test_results.csv` is a probability between 0 and 1 that the crystal can be classified as 1 (metal in the above example).

After predicting, you will get one file in `cgcnn` directory:

- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set. Here the target value is just any number that you set while defining the dataset in `id_prop.csv`, which is not important.

## Data

To reproduce our paper, you can download the corresponding datasets following the [instruction](data/material-data).

## Authors

This software was primarily written by [Tian Xie](http://txie.me) who was advised by [Prof. Jeffrey Grossman](https://dmse.mit.edu/faculty/profile/grossman). 

## License

CGCNN is released under the MIT License.



