# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:31:06 2021

@author: @Markovien
"""
import os, rdkit, urllib, gzip, shutil
from rdkit import Chem

base = os.getcwd()
output_folder = base + '/data/'

def download_chebi():
    loc = base + "/ChEBI_complete_3star.sdf.gz" #le chemin absolu vers ton fichier en local
    url = "ftp://ftp.ebi.ac.uk/pub/databases/chebi/SDF/ChEBI_complete_3star.sdf.gz" #le lien de téléchargement
    if not os.path.exists(loc):
        urllib.request.urlretrieve(url, loc)
    else: pass

    #on décompresse le fichier téléchargé avec ça
    new_file = base + "/ChEBI_complete_3star.sdf" #le chemin absolu vers ton fichier décompressé en local
    with gzip.open(loc, 'rb') as file_in:
        with open(new_file, 'wb') as file_out:
            shutil.copyfileobj(file_in, file_out)
    return new_file
            
def create_folder():

    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

def parsing():
    
    create_folder()
    i = 0
    if not os.path.exists(os.path.join(os.getcwd(), 'ChEBI_complete_3star.sdf')):
        file = download_chebi()
    else:
        file = os.path.join(os.getcwd(), 'ChEBI_complete_3star.sdf')

    location = output_folder + 'chebi_smiles.txt'
    outf = open(location, 'w+')
    suppl = Chem.SDMolSupplier(file, sanitize = False) #Lecture fichier sdf et transformation en tableau de mol
    
    for mol in suppl:

        try:
            mol.UpdatePropertyCache(strict=False)
            mol = Chem.RemoveAllHs(mol, sanitize=False)
            
            if mol is not None:
                tmp = str(Chem.MolToSmiles(mol))
            
                #Ecriture dans le fichier
                outf.write(tmp + '\n')
                i = i + 1
        except:
            pass
        
    #Ecriture fichier molécules non lues
    print("Il y a {} molécules dans le fichier".format(i))
    outf.close()


parsing()
