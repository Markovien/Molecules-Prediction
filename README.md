# RNN program for molecules prediction with Pytorch

the purpose of this project is to train a RNN with Pytorch in order to predict the SMILES description of molecule.
We don't care about the chemistry exactness of the predicted SMILES, we just want to predict something which can be considered as a molecule (verification with RDKit); this open the door for ameliorations.

# How to use

- First: put all python codes in the same folder. Let call this folder "main".
- Second: if you want to train your RNN with ChEBI 3 stars molecules, run download_data_set.py.
        Else, create a new folder in main and call it "data". then, put inside chembl_smiles.txt.
- Third: run data_processing.py
- Fourth: run generator_training.py
- Finally: test your trained RNN with generator_test.py

# Good to know

We are using Pytorch with CPU-only and RDKit. Make sure that they are alredy installed on your computer.
