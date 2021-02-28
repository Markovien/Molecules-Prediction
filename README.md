# RNN program for generating (novel) molecules with Pytorch

the purpose of this project is to train a RNN with Pytorch in order to generate the SMILES description of molecule.
We don't care about the chemistry exactness of the generated SMILES, we just want to generate something molecule like (verification with RDKit); this open the door for ameliorations.

# How to use

- First: put all python codes in the same folder. Let call this folder "main".
- Second: if you want to train your RNN with ChEBI 3 stars molecules, run download_data_set.py.
        Else, create a new folder in main and call it "data". then, put inside chembl_smiles.txt which is available on : https://drive.google.com/file/d/1gXGxazJDIhjlGFwOCt8J_BET7qbVSDZ_/view?usp=sharing.
- Third: run data_processing.py
- Fourth: run generator_training.py
- Finally: test your trained RNN with generator_test.py

# Good to know

We are using Pytorch with CPU-only and RDKit. Make sure that they are alredy installed on your computer.

**Thanks to : @LamUong**
