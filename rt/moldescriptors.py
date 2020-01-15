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
from getf import getf

def read_lib_identifiers(infile):
    """
    Get the identifiers present in the library
    
    Parameters
    ----------
    infile : str
        infile name that contains previously analyzed structures to speed up the process

    Returns
    -------
    set
        structures present in the library
    """
    infile = open(infile)
    identifiers = []
    for line in infile:
        if line.startswith("IDENTIFIER"): continue
        line = line.strip()
        identifiers.append(line.split(",")[0])
    return(set(identifiers))

def get_feats_lib(infile,
                  extract_list):
    """
    Get the features for the library to speed up the process
    
    Parameters
    ----------
    infile : str
        infile name that contains previously analyzed structures to speed up the process
    extract_list : list
        extract certain structures from the library

    Returns
    -------
    dict
        dictionary with molecular descriptors
    """
    infile = open(infile)
    ret_dict = {}
    unique_smiles = set([j for i,j in extract_list])
    smiles_to_name = dict([(j,i) for i,j in extract_list])
    for line in infile:
        if line.startswith("IDENTIFIER"): header = line.strip().split(",")
        line = line.rstrip()
        identifier = line.split(",")[0]
        if identifier in unique_smiles:
            split_line = line.split(",")[1:]
            ret_dict[smiles_to_name[identifier]] = dict(zip(header[1:],split_line))
    return(ret_dict)

def get_features(infile_name="data/LMSDFDownload28Jun15FinalAll.sdf",
                 outfile_name="lmfeatures.csv",
                 library_file="feats_lib.csv",
                 id_index=0,
                 mol_index=1,
                 time_index=None,
                 gui_object=None,
                 chunk_size=10000):
    """
    Get molecular descriptors for a dataset
    
    Parameters
    ----------
    infile_name : str
        infile name that contains the structures (inchi or smiles)
    outfile_name : str
        outfile name for structures linked to molecular descriptors
    library_file : str
        infile name that contains previously analyzed structures to speed up the process
    id_index : int
        index number of identifier
    mol_index : int
        index number of the inchi or smiles
    gui_object : object
        qt object of the GUI to update the process
    chunk_size : int
        number of analytes to analyze

    Returns
    -------

    """
    outfile = open(outfile_name,"w")

    features = []
    library_analyze_smiles = []
    lipid_dict = {}
    time_dict = {}
    counter = 0
    write_header = True

    if gui_object: tot_to_analyze = float(len(open(infile_name).readlines()))

    if len(library_file.strip()) > 0: lib_identifiers = read_lib_identifiers(library_file)
    else: lib_identifiers = set()

    mols = open(infile_name)

    
    for mol in mols:
        counter += 1

        if gui_object:
            perc = round((float(counter)/tot_to_analyze)*100.0,1)
            if perc.is_integer():
                gui_object.update_progress2(perc)

        if (counter % chunk_size) == 0 and counter != 0:
            if len(library_analyze_smiles) > 0: 
                lib_dict = get_feats_lib(library_file,library_analyze_smiles)
                lipid_dict.update(lib_dict)

            if len(features) == 0:
                features = list(lipid_dict[list(lipid_dict.keys())[0]].keys())

            if write_header:
                if time_index: outfile.write("IDENTIFIER,time,%s\n" % (",".join(features)))
                else: outfile.write("IDENTIFIER,%s\n" % (",".join(features)))
                write_header = False

            for identifier in lipid_dict.keys():
                if time_index:
                    outfile.write("%s," % (identifier))
                    outfile.write("%s" % (time_dict[identifier]))
                else: outfile.write("%s" % (identifier))
                for f in features:
                    outfile.write(",%s" % (lipid_dict[identifier][f]))
                outfile.write("\n")
            #outfile.close()
            library_analyze_smiles = []
            lipid_dict = {}
            time_dict = {}

        if "\t" in mol: mol = mol.strip().split("\t")
        else: mol = mol.strip().split(",")

        identifier = mol[id_index]
        mol_str = mol[mol_index]
        if mol_str == "SMILES": continue
        if time_index: 
            rt = mol[time_index]
            time_dict[identifier] = rt

        m = Chem.MolFromSmiles(mol_str)

        if m == None: 
            # TODO write error msg
            continue
        if mol_str not in lib_identifiers:
            try: fdict = getf(m)
            except: continue
            lipid_dict[identifier] = fdict["rdkit"]
            if len(features) == 0: features = list(set(lipid_dict[identifier].keys()))
        else:
            library_analyze_smiles.append([identifier,mol_str])
    
    if len(library_analyze_smiles) > 0: 
        lib_dict = get_feats_lib(library_file,library_analyze_smiles)
        lipid_dict.update(lib_dict)

    if len(features) == 0:
        features = list(lipid_dict[list(lipid_dict.keys())[0]].keys())

    if write_header:
        if time_index: outfile.write("IDENTIFIER,time,%s\n" % (",".join(features)))
        else: outfile.write("IDENTIFIER,%s\n" % (",".join(features)))
        write_header = False

    for identifier in lipid_dict.keys():
        
        if time_index:
            outfile.write("%s," % (identifier))
            outfile.write("%s" % (time_dict[identifier]))

        else: outfile.write("%s" % (identifier))
        for f in features:
            outfile.write(",%s" % (lipid_dict[identifier][f]))
        outfile.write("\n")
    outfile.close()


if __name__ == "__main__":
    get_features()