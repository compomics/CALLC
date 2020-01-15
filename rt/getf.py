"""
Robbin Bouwmeester

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
This code is used to train retention time predictors and store
predictions from a CV procedure for further analysis.

This project was made possible by MASSTRPLAN. MASSTRPLAN received funding 
from the Marie Sklodowska-Curie EU Framework for Research and Innovation 
Horizon 2020, under Grant Agreement No. 675132.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from subprocess import Popen
from subprocess import PIPE
from os import remove

def rdkit_descriptors(mol):
    """
    Get the rdkit descriptors
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object from rdkit
    
    Returns
    -------
    dict
        feature to molecular descriptor dictionary
    """
    ret_dict = {}

    # Iterate over all molecular descriptors
    for name,func in Descriptors.descList:
        ret_dict[name] = func(mol)
    return(ret_dict)

def cdk_descriptors(mol,temp_f_smiles_name="tempsmiles.smi",temp_f_cdk_name="tempcdk.txt"):
    """
    Get the cdk descriptors
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object from rdkit
    temp_f_smiles_name : str
        temporary file for storing the smile
    temp_f_cdk_name : str
        temporary file for storing the cdk output
    
    Returns
    -------
    dict
        feature to molecular descriptor dictionary
    """
    ret_dict = {}

    smiles = Chem.MolToSmiles(mol,1)
    
    temp_f_smiles = open(temp_f_smiles_name,"w")
    temp_f_smiles.write("%s temp" % smiles)
    temp_f_smiles.close()

    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="topological"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="geometric"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="constitutional"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="electronic"))
    ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="hybrid"))

    remove(temp_f_smiles_name)
    remove(temp_f_cdk_name)

    return(ret_dict)


def call_cdk(infile="",outfile="",descriptors=""):
    """
    Call to getting cdk descriptors
    
    Parameters
    ----------
    descriptors : 
        name of the descriptors
    infile : str
        file for storing the smile
    outfile : str
        file for storing the cdk output
    
    Returns
    -------
    dict
        feature to molecular descriptor dictionary
    """
    cmd = "java -jar CDKDescUI-1.4.6.jar -b %s -a -t %s -o %s" % (infile,descriptors,outfile)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out = p.communicate()
    return(parse_cdk_file(outfile))

def parse_cdk_file(file):
    """
    Parse the output of cdk to a dictionary
    
    Parameters
    ----------
    file : str
        file to parse from cdk
    
    Returns
    -------
    dict
        feature to molecular descriptor dictionary
    """
    cdk_file = open(file).readlines()
    cols = cdk_file[0].strip().split()[1:]
    feats = cdk_file[1].strip().split()[1:]
    return(dict(zip(cols, feats)))

def getf(mol,progs=["rdkit"]):
    """
    Get molecular descriptors for a molecule
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object from rdkit
    progs : list
        choose either rdkit and/or cdk
    
    Returns
    -------
    dict
        feature to molecular descriptor dictionary
    """
    ret_dict = {}
    if "rdkit" in progs: ret_dict["rdkit"] = rdkit_descriptors(mol)
    if "cdk" in progs: ret_dict["cdk"] = cdk_descriptors(mol)
    return(ret_dict)


if __name__ == "__main__":
    test_smile = "N12CCC36C1CC(C(C2)=CCOC4CC5=O)C4C3N5c7ccccc76"
    test_mol = Chem.MolFromSmiles(test_smile)
    get_features(test_mol)