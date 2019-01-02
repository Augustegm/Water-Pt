# Water-Pt

This repository contains scripts that use a feedforward neural network to fit forces for a water-Pt(111) interface. 

The repository currently consists of 3 files:

1) dataset.traj contains the first 1000 configurations of the data set described in https://www.doi.org/10.1039/C8SC02495B, 
which is a large DFT data set of the water-Pt(111) calculated using ab initio molecular dynamics. The reason for using only a small fraction
of the original data set is due to the file size constrains of GitHub. 

2) fingerprint.py is a script that can be used to fingerprint the atomic environment of each atom in the data set based on the
formalism described in https://www.doi.org/10.1021/acs.jpcc.6b10908

3) Neuralnet.py is the central script that sets up the training and data set, calculates the fingerprints using fingerprint.py 
and then constructs and fits a feedforward neural network to the data using PyTorch. Various option are given for the optimization
and structure of the neural network. 
