# aneurysm-segmentation


Do 3D Deep learning Segmentation on 3D-Point data to predict WSS values for Intracranial Aneurysms


The Project is divided into three main parts:
1) Conf - Contains config.yaml files for all the sttings
2) notebooks - Contains EDA and some pipleine testing
3) Script - Contains all the codes for training of the DL network.

Scripts are divided into following modules:
3.1) Dataset: Dataset builder script for creating custom dataset loaders
3.2) Features: Contains feature maker script which includes preprocessing of meshes, calculation of curvatures, calculation of 3D local descriptors, Downsampling WSS values and final concatenation of all the features to make a downsampled mesh.ply file
3.3) Utils: General utility functions
3.4) Visualization: Script for creating data loaders for forward pass
3.5) Modelling: Has scripts for creating a model, a trainer and initiating a training. 


To start training, 
1) A config file has to be defined that resides in ../conf/ path. 
2) Path of the config has to be mentioned in the ../scripts/modelling/train_chkp.py
3) train_chkp.py is run, which uses trainer_chkp.py and dataset created from ../scripts/dataset/AneurysmDataset.py to load the dataset and a model into a model checkpoint and start the training process.


A config file has to be defined by keeping following points in the mind for different experimentations:
1) parts_to_segment: number of classes
2) features_to_include: Which features should be there in the dataset
3) class_weight_type: custom (user defined) or automatic (based on point counts for each class)
4) scaler_type: global or local scaling
5) wandb.log: true or false for online tracking
6) training.epochs: Number of epochs to run


The dataset file has to be specifically placed in the folder with path ../aneurysm_segmentation3d/datasets/data/aneurysm/raw/ . This is to ensure that the AneurysmDataset.py file finds the data at aneurysm/raw. Path "/aneurysm_segmentation3d/datasets/data/" is a requirement by torchpoint3d to find the files. 

