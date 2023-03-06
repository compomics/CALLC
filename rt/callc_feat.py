"""
This code can be used to extract features for a model to
predict CCS values. For an example on how to use, see
the ccs_main_train.py script.
"""

__author__ = "Robbin Bouwmeester"
__credits__ = ["Robbin Bouwmeester", "Waters", "Hans Vissers"]
__version__ = "1.0"
__maintainer__ = "Robbin Bouwmeester"
__email__ = "Robbin.bouwmeester@ugent.be"

# Library imports
from getf import getf

# Native imports
import logging
from copy import deepcopy
import pickle

# Pandas
import pandas as pd

# RDKit
import rdkit
from rdkit import Chem

# ML imports
from sklearn.preprocessing import StandardScaler

# Data imports
import numpy as np

from io import StringIO

from tqdm import tqdm

def get_feats(data_df: pd.DataFrame,
              RDKit_r: bool = True,
              cdk: bool     = False,
              mordred: bool = True) -> pd.DataFrame:
    """
    Function to get features (chemical descriptors and chemical class)
    from a csv and put them in a pandas dataframe.

    Parameters
    ----------
    infile : str
        csv file that contains the CCS data to fit the model
    ions : list
        ions to include for the feature extraction
    include_class_vis : list
        chemical classes to include for feature extraction

    Returns
    -------
    object :: pd.DataFrame
        dataframe containing the chemical descriptors
    """
    #data_df = pd.read_csv(infile, encoding="latin1")
    #data_df["InChI Code"] = data_df["InChI Code"].fillna("")
    if type(data_df) == str:
        if "\t" in str(data_df):
            data_df = pd.read_csv(StringIO(data_df),sep="\t")
        else:
            data_df = pd.read_csv(StringIO(data_df))
        
    data_df["index"] = list(range(len(data_df.index)))

    feat_dict = {}

    idents = []
    mols = []

    print("Reading data:")
    for row_index, row in tqdm(data_df.iterrows()):
        inchi_code, _ = list(row[["inchi", "index"]])

        # InCHI is empty... so skip
        if len(inchi_code) == 0:
            logging.error("Going to skip the following line due to missing InCHI: %s" %
                          (row_index))
            continue

        # Make an identifier based on the index (number) and the chemical name
        identifier = str(int(row["index"]))+"|"+str(row["IDENTIFIER"])
        # Get the features
        #try:
        if inchi_code.startswith("InChI"):
            mol_obj = rdkit.Chem.inchi.MolFromInchi(inchi_code)
        else:
            mol_obj = Chem.MolFromSmiles(inchi_code)

        if mol_obj == None:
            continue

        #feat_dict[identifier] = getf(mol_obj)["rdkit"]
        feat_dict[identifier] = {} #getf(mol_obj,progs=["mordred"])["mordred"]
        mols.append(mol_obj)
        idents.append(identifier)

        # Provide the features (chemical descriptors) for each ion
        #for ion in ions:
        #    feat_dict[identifier][ion] = row[ion]

        #except BaseException as e:
        #    logging.error("Could not extract features for: %s" %
        #                  (row))
        #    logging.error("For this row got the following error: %s" %
        #                  (e))
        #    continue

    mol_desc = {}
    if RDKit_r:
        mol_desc_rdkit = getf(mols,progs=["rdkit"])["rdkit"]
        print("Running rdkit:")
        for k in tqdm(mol_desc_rdkit.keys()):
            try:
                mol_desc[k].update(mol_desc_rdkit[k])
            except:
                mol_desc[k] = {}
                mol_desc[k].update(mol_desc_rdkit[k])
    if mordred:
        mol_desc_mordred = getf(mols,progs=["mordred"])["mordred"]
        print("Running mordred:")
        for k in tqdm(mol_desc_mordred.keys()):
            try:
                mol_desc[k].update(mol_desc_mordred[k])
            except:
                mol_desc[k] = {}
                mol_desc[k].update(mol_desc_mordred[k])
    if cdk:
        mol_desc_cdk = getf(mols,progs=["cdk"])["cdk"]
        for k in mol_desc_cdk.keys():
            try:
                mol_desc[k].update(mol_desc_cdk[k])
            except:
                mol_desc[k] = {}
                mol_desc[k].update(mol_desc_cdk[k])

    #except BaseException as e:
    #    logging.error("Could not extract features for: %s" %
    #                    (row))
    #    logging.error("For this row got the following error: %s" %
    #                    (e))
    for i, ident in enumerate(idents):
        #temp_dict = {[list((k,v)) for k,v in mol_desc[i].items() if type(v) != object]}
        if i in mol_desc.keys() or str(i) in mol_desc.keys():
            try:
                for k,v in mol_desc[i].items():
                    try:
                        v = float(v)
                    except:
                        continue
                    if type(v) == float or type(v) == int:
                        feat_dict[ident][k] = mol_desc[i][k]
            except KeyError:
                pass
            try:
                for k,v in mol_desc[str(i)].items():
                    print(k,v,i)
                    try:
                        v = float(v)
                    except:
                        continue
                    if type(v) == float or type(v) == int:
                        feat_dict[ident][k] = mol_desc[str(i)][k]
                    else:
                        pass
            except KeyError:
                pass
        feat_dict[ident]["IDENTIFIER"] = data_df.iloc[int(ident.split("|")[0]),:]["IDENTIFIER"]
        try:
            feat_dict[ident]["system"] = data_df.iloc[int(ident.split("|")[0]),:]["system"]
        except:
            pass
        try:
            feat_dict[ident]["time"] = data_df.iloc[int(ident.split("|")[0]),:]["time"]
        except:
            pass


        #feat_dict[ident]
        #feat_dict[ident] = {**mol_desc[i], **feat_dict[ident]}

#    mol_desc["IDENTIFIER"] = row["IDENTIFIER"]
#    try:
#        mol_desc["time"] = row["time"]
#    except:
#        pass
#    try:
#        mol_desc["system"] = row["system"]
#    except:
#        pass

    data_pd = pd.DataFrame(feat_dict).transpose()

    return data_pd


def data_pd_to_feat(data,
                    ions=["CCS (+H)"],
                    scale=True,
                    sel_features=["MolWt",
                                  "LabuteASA"],
                    vis_class=None,
                    scaler_file="models/scaler.pickle"):
    """
    Function to go from a pandas dataframe (chemical descriptors) to
    a matrix shape that can be used as input for training a ML model.
    In addition, it remove any strange characters ("," -> "_") and
    removes any duplicated molecules based on the chemical descriptors.

    Parameters
    ----------
    data : object :: pd.DataFrame
        pandas dataframe containing chemical descriptors
    ions : list
        ions to include for the feature extraction
    scale : Boolean
        flag to scale the features in the X-matrix
    vis_class : object :: pd.DataFrame
        pandas dataframe containing the chemical classes
    scaler_file : str
        the output file for the scaler

    Returns
    -------
    object :: pd.DataFrame
        matrix with all chemical descriptors
    object :: pd.DataFrame
        vector with all objective values (CCS)
    object :: StandardScaler
        scaler used to fit and transform the input data
    list
        list of feature names
    object :: pd.DataFrame
        dataframe containing all chemical classes
    """
    feat_df = data.drop(ions, axis=1)

    to_aggregate_list = []
    to_aggregate_list_ccs = []

    feat_names = deepcopy(ions)
    feat_names.extend(data.columns)

    # For each analyte go over all possible adducts
    for ion in ions:
        df_slice = feat_df[~data[ion].isna()]
        df_slice[ion] = len(df_slice.index)*[1]

        # Go from class (string) to one-hot-encoding
        transl_classes_to_dummys = pd.get_dummies(vis_class.loc[df_slice.index,
                                                                "Superclass"])

        df_slice = pd.concat([df_slice, transl_classes_to_dummys],
                             axis=1,
                             join="inner")

        df_slice.index = [ion+"|"+index_name for index_name in df_slice.index]

        to_aggregate_list.append(df_slice)

        y_slice = data[~data[ion].isna()][ion]
        y_slice.index = df_slice.index
        to_aggregate_list_ccs.append(y_slice)

    # Create the dataframes needed for training (X matrix and y vector)
    X = pd.concat(to_aggregate_list).fillna(0)
    y = pd.concat(to_aggregate_list_ccs)

    if len(sel_features) > 0:
        sel_features.extend(ions)
        X = X[sel_features]

    # Apply scaler if required
    scaler = StandardScaler()
    if scale:
        scaler = StandardScaler()
        scaler.fit_transform(X)
        X[X.columns] = scaler.fit_transform(X[X.columns])

    with open(scaler_file, "wb") as handle:
        pickle.dump(scaler, handle)

    class_feats = [f for f in X.columns if f not in feat_names]

    # Apply sanitation step on names and drop duplicate analytes
    X.index = [x_i.replace(",", "_") for x_i in X.index]
    y.index = [y_i.replace(",", "_") for y_i in X.index]
    X = X.drop_duplicates(subset=None, keep="first", inplace=False)
    y = y[X.index]

    # Drop those analytes without CCS
    rem_indexes = X[y.isna()].index
    X.drop(rem_indexes, inplace=True)
    y.drop(rem_indexes, inplace=True)

    return X, y, scaler, feat_names, class_feats

if __name__ == "__main__":
    dict_inchi = {"InChI Code": ["InChI=1S/C17H19NO3/c1-18-7-6-17-10-3-5-13(20)16(17)21-15-12(19)4-2-9(14(15)17)8-11(10)18/h2-5,10-11,13,16,19-20H,6-8H2,1H3/t10-,11+,13-,16-,17-/m0/s1", 
                        "InChI=1S/C17H19NO3/c1-18-7-6-17-10-3-5-13(20)16(17)21-15-12(19)4-2-9(14(15)17)8-11(10)18/h2-5,10-11,13,16,19-20H,6-8H2,1H3/t10-,11+,13-,16-,17-/m0/s1",
                        "InChI=1S/C17H19NO3/c1-18-7-6-17-10-3-5-13(20)16(17)21-15-12(19)4-2-9(14(15)17)8-11(10)18/h2-5,10-11,13,16,19-20H,6-8H2,1H3/t10-,11+,13-,16-,17-/m0/s1",
                        "InChI=1S/C17H19NO3/c1-18-7-6-17-10-3-5-13(20)16(17)21-15-12(19)4-2-9(14(15)17)8-11(10)18/h2-5,10-11,13,16,19-20H,6-8H2,1H3/t10-,11+,13-,16-,17-/m0/s1"],
                  "index": ['1', '2', '3', '4'],
                  "Chemical Name" : ["a","b","c","d"]}
    df_inchi = pd.DataFrame(dict_inchi)
    print(get_feats(df_inchi))