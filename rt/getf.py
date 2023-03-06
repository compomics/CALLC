"""
This code can be used to extract chemical descriptors
for SMILES or InCHI structures. The packages used to
extract the features can be RDKit and CDK.
"""

# Native imports
from subprocess import Popen
from subprocess import PIPE

from os import remove

import logging

# TODO cleanup
#import tensorflow as tf
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import Draw
import hashlib
#from rdkit.Chem.Draw import IPythonConsole
#from model import MoleculeVAE
#from utils import encode_smiles, decode_latent_molecule, interpolate, get_unique_mols

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors

# mordred
from mordred import Calculator, descriptors
import pandas as pd
import time
#from obabel_wrapper_batch import get_feat

from tqdm import tqdm

def rdkit_descriptors(mol):
    """
    Function to get (all) chemical descriptors from RDKit

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    ret_dict = {}

    # Iterate over all functions that retrieve chemical descriptors
    for name, func in Descriptors.descList:
        ret_dict[name+"_rdkit"] = func(mol)

    return(ret_dict)


def cdk_descriptors(mol,
                    temp_f_smiles_name="tempsmiles.smi",
                    temp_f_cdk_name="tempcdk.txt"):
    """
    Function to get chemical descriptors from CDK

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    ret_dict = {}

    smiles = Chem.MolToSmiles(mol, 1)

    # Write temporary input file for analysis
    temp_f_smiles = open(temp_f_smiles_name, "w")
    temp_f_smiles.write("%s temp" % smiles)
    temp_f_smiles.close()

    # Get multiple groups of chemical descriptors
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,
                             outfile=temp_f_cdk_name,
                             descriptors="topological"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,
                             outfile=temp_f_cdk_name,
                             descriptors="geometric"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,
                             outfile=temp_f_cdk_name,
                             descriptors="constitutional"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,
                             outfile=temp_f_cdk_name,
                             descriptors="electronic"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,
                             outfile=temp_f_cdk_name,
                             descriptors="hybrid"))

    # Remove temp files
    remove(temp_f_smiles_name)
    remove(temp_f_cdk_name)

    return(ret_dict)

def call_rcdk(infile_name="temp/tempKnownsl2.csv",outfile_name="temp/tempUnknownsl2.csv"):
    """
    Get the dataframe associated with this analysis
    
    Parameters
    ----------
    infile_known : str
        location of a file with known retention time, for Layer 2
    infile_unknown : str
        location of a file with umknown retention time, for Layer 2
    fold_list : str
        the folds to be used in Layer 2
    
    Returns
    -------
    pd.DataFrame
        test predictions
	pd.DataFrame
		train predictions
    """
    # Write enumerate(ID)+smiles to file -> read file -> execute CDK -> read file here -> todict() -> return
    cmd = "Rscript CDK_run.R %s %s" % (infile_name,outfile_name)
    print(cmd)
    p = Popen(cmd, stdout=PIPE,stderr=PIPE,shell=True)
    out, err = p.communicate()

    print(out)
    print(err)

    preds = pd.read_csv(outfile_name)
    preds.index = preds["ident"]

    return preds[preds.columns.difference(["ident","SMILES"])].T.to_dict()

def call_cdk(infile="",
             outfile="",
             descriptors=""):
    """
    Function to make the call to CDK for chemical descriptor calculation

    Parameters
    ----------
    infile : str
        file to parse the chemical descriptors from
    outfile : str
        temporary out file
    descriptors : str
        type of descriptors to calculate

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    cmd = "java -jar CDKDescUI-1.4.6.jar -b %s -a -t %s -o %s" % (infile,
                                                                  descriptors,
                                                                  outfile)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)

    # Wait for results ...
    out = p.communicate()

    logging.info(out)

    return(parse_cdk_file(outfile))


def parse_cdk_file(file):
    """
    Function to parse CDK output

    Parameters
    ----------
    file : str
        file to parse the chemical descriptors from

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    cdk_file = open(file).readlines()
    cols = cdk_file[0].strip().split()[1:]
    feats = cdk_file[1].strip().split()[1:]
    return(dict(zip(cols, feats)))

def mordred_descriptors_3D(mol):
    df = pd.DataFrame(mol,index=list(range(len(mol))),columns=["InChI Code"])
    df["index"] = list(range(len(mol)))
    df["InChI Code"] = [str(Chem.MolToSmiles(m)) for m in df["InChI Code"]]
    df_feat = get_feat(df,n_jobs=256)
    return df_feat.to_dict()

def mordred_descriptors(mol):
    """
    Function to get chemical descriptors from CDK

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    calc = Calculator(descriptors, ignore_3D=True)
    if type(mol) == list:
        df = calc.pandas(mol,nproc=1).T
        return df.to_dict()
    else:
        df = calc.pandas([mol],nproc=1).T
        return df.to_dict()[0]

def rcdk_descriptors(mols,temp_file_name="temp_smiles.csv",result_file_name="cdk_features.csv"):
    temp_file = open(temp_file_name,"w")
    temp_file.write("ident,SMILES\n")
    for i,mol in enumerate(mols):
        temp_file.write("%s,%s\n" % (i,Chem.MolToSmiles(mol, 1)))
    temp_file.close()

    # parallelize from here on out
    return call_rcdk(infile_name=temp_file_name,outfile_name=result_file_name)

def autoencoder_descriptors(mols):
    # TODO move to parameters
    trained_model = 'chembl_23_model.h5'
    charset_file = 'charset.json'
    latent_dim = 292
    feat_names = ["autoencoder_lat_"+str(i) for i in range(latent_dim)]

    ret_dict = {}

    # load charset and model
    with open('charset.json', 'r') as outfile:
        charset = json.load(outfile)

    model = MoleculeVAE()
    model.load(charset, trained_model, latent_rep_size = latent_dim)

    for i,mol in enumerate(mols):
        smiles = Chem.MolToSmiles(mol, 1)
        if len(smiles) > 120:
            smiles = smiles[0:119]
        
        ret_dict[i] = dict(zip(feat_names,encode_smiles(smiles, model, charset)[0]))
    
    return ret_dict

def getf(mol, progs=["rdkit"]):
    """
    Main Function to get chemical descriptors from RDKit and/or CDK

    Parameters
    ----------
    mol : object :: rdkit.Chem.rdchem.Mol
        mol object from rdkit
    progs : list
        list of programs to run (rdkit or cdk)

    Returns
    -------
    dict
        dictionary containing the chemical descriptor name and
        the chemical descriptor value
    """
    ret_dict = {}
    if "rdkit" in progs:
        if type(mol) == list:
            ret_dict["rdkit"] = {}
            print("Calculating rdkit:")
            for i,m in tqdm(enumerate(mol)):
                ret_dict["rdkit"][i] = rdkit_descriptors(m)
        else:
            ret_dict["rdkit"] = rdkit_descriptors(mol)
    if "cdk" in progs:
        temp_file_name="temp_smiles.csv"
        result_file_name="cdk_features.csv"
        
        ms = str(int(time.time_ns()))
        
        hash_object = hashlib.sha1(ms.encode())
        hex_dig = hash_object.hexdigest()
        hex_dig_in = "temp/"+hex_dig+"_temp_smiles.csv"
        hex_dig_out = "temp/"+hex_dig+"_cdk_features.csv"

        ret_dict["cdk"] = rcdk_descriptors(mol,temp_file_name=hex_dig_in,result_file_name=hex_dig_out)
    if "mordred" in progs:
        ret_dict["mordred"] = mordred_descriptors(mol)
    if "mordred3D" in progs:
        ret_dict["mordred3D"] = mordred_descriptors_3D(mol)
    if "autoencoder" in progs:
        ret_dict["autoencoder"] = autoencoder_descriptors(mol)
    return(ret_dict)


if __name__ == "__main__":
    test_smiles = ["CCCCC","CCC","C"]
    test_mols = [Chem.MolFromSmiles(ts) for ts in test_smiles]
    res = getf(test_mols,progs=["mordred3D"])
    print(res)

