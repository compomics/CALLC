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

import copy

import subprocess

from random import shuffle
from sklearn.model_selection import KFold

import pandas as pd

from trainl1 import train_l1_func
from applyl1 import apply_models
from trainl2_pygam import apply_l2
from trainl3 import train_l3

from callc_feat import get_feats
from sklearn.preprocessing import StandardScaler

from pickle import dump
from pickle import load

import numpy as np
from numpy import median

import os
import time

import hashlib

def move_models(k):
    """
    Move models so they will not be used in Layer 1
    
    Parameters
    ----------
    k : str
		key name for the models that need to be moved

    Returns
    -------

    """
    cmd = "mv mods_l1/%s*.pickle mods_l1/temp/" % (k)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    p.communicate()


def remove_models(k,n):
    """
    Remove specific models
    
    Parameters
    ----------
    k : str
		key name for the models that need to be moved
	n : str
		specific numeric identifier for the model to be remove

    Returns
    -------

    """
    cmd = "rm -rf mods_l1/%s*.pickle" % (k)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    p.communicate()

def move_models_back(k):
    """
    Move models back so they will be used in Layer 1
    
    Parameters
    ----------
    k : str
		key name for the models that need to be moved

    Returns
    -------

    """
    cmd = "mv mods_l1/temp/%s*.pickle mods_l1/" % (k)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    p.communicate()

def cv_to_fold(cv,num_ins):
    """
    Define a CV in a pre-defined list
    
    Parameters
    ----------
    cv : sklearn.model_selection.KFold
		cv to be put into the list
	num_ins : int
		number of folds

    Returns
    -------
	list
		defined cv
    """
    ret_vec = [0]*num_ins
    counter_f = 0
    for train,test in cv:
        for t in test:
            ret_vec[t] = counter_f
        counter_f += 1
    return(ret_vec)

def calc_overlap_compounds(compounds):
    tot_compounds = len(compounds)
    compounds = sorted(compounds)

    optimal_dist = 1.0/tot_compounds

    all_dists = []
    for i,c in enumerate(compounds):
        if i == len(compounds)-1:
            continue
        dist = (optimal_dist+c)-compounds[i+1]

        if dist < 0.0:
            all_dists.append(0.0)
        else:
            all_dists.append(dist)
        
    return sum(all_dists)


def replace_non_ascii(ident):
	try:
		return "".join([str(s) for s in ident if s.isascii()])
	except TypeError:
		try:
			ident = str(ident)
			return "".join([str(s) for s in ident if s.isascii()])
		except TypeError:
			return "Non-ident" 

def make_preds(reference_infile="train_set_lpp2.csv",pred_infile="lmfeatures.csv",k="CALLCtemp",outfile="",extra_pred_file="",outfile_modname="",num_jobs=4,GUI_obj=None,ch_size=100000):
    """
    Make predictions for the evaluation of CALLC
    
    Parameters
    ----------
    reference_infile : str
		location of train data
    pred_infile : str
        location of file to make predictions for
    k : str
        key name to add to predictions and models
    outfile : str
        outfile for the predictions
    outfile_modname : str
        name for the models it will train
    num_jobs : int
        number of threads to spawn
    GUI_obj : object
        gui object to update log
    ch_size : int
        chunk size for generating predictions

    Returns
    -------

    """
    try: os.chdir("rt/")
    except: pass
    
    if type(reference_infile) == str: 
        ref_infile = pd.read_csv(reference_infile)
    else:
        try:
            ref_infile = get_feats("".join([l for l in reference_infile])) #.decode()
        except TypeError:
            ref_infile = get_feats("".join([l.decode() for l in reference_infile])) #

    
    ref_infile["IDENTIFIER"] = ref_infile["IDENTIFIER"].apply(replace_non_ascii)
    ref_infile.to_csv("Degradation_6_with_smiles_feats.csv")
    # Make sure we have the correct data types
    dict_dtypes = dict(ref_infile.select_dtypes(include=['int']).apply(pd.to_numeric,downcast="integer").dtypes)
    float_dtypes = dict(ref_infile.select_dtypes(include=['float']).apply(pd.to_numeric,downcast="float").dtypes)
    dict_dtypes.update(float_dtypes)

    if type(reference_infile) == str:
        tot_preds = sum(1 for row in open(pred_infile,"r"))/ch_size
        p_infile = pd.read_csv(pred_infile,dtype=dict_dtypes,chunksize=ch_size)
    else:
        try:
            p_infile = get_feats("".join([l for l in pred_infile])) #.decode()
        except TypeError:
            p_infile = get_feats("".join([l.decode() for l in pred_infile]))
    
    p_infile.to_csv("Degradation_6_with_smiles_feats.csv")
    infile = pd.read_csv("datasets/input_for_scaler.csv",low_memory=False)	
    infile.fillna(0.0,inplace=True)

    try:
        keep_f = [x.strip() for x in open("features/selected_features_v3.txt", encoding="utf-8").readlines()]
        infile = infile[keep_f]
        keep_f_features_only = [f for f in keep_f if f not in ["time","IDENTIFIER","system"]]
        infile[keep_f_features_only] = infile[keep_f_features_only].applymap(lambda x: np.nan if isinstance(x, str) else x)
        infile.fillna(0.0,inplace=True)
    except IOError:
        infile,keep_f = sel_features(infile)
        outfile = open("features/selected_features_v3.txt","w")
        outfile.write("\n".join(list(keep_f)))
        
        keep_f_features_only = [f for f in keep_f if f not in ["time","IDENTIFIER","system"]]        
        infile[keep_f_features_only] = infile[keep_f_features_only].applymap(lambda x: np.nan if isinstance(x, str) else x)
        infile.fillna(0.0,inplace=True)

    n = len(ref_infile.index)

    print("===========")
    print("Total number of train molecules with tR: %s" % (n))

    keep_f_withoutid = list(keep_f_features_only)

    scaler = StandardScaler()
    
    infile.fillna(0.0,inplace=True)
    scaler.fit_transform(ref_infile[keep_f_withoutid])
    #scaler = load(open('scaler.pkl', 'rb'))

    ref_infile[keep_f_withoutid] = scaler.transform(ref_infile[keep_f_withoutid])

    # Make sure that for the training data we do not have infinite or nan
    train = ref_infile
    
    # Define the folds to make predictions
    cv = KFold(n_splits=10,shuffle=True,random_state=42)
    cv = list(cv.split(train.index))
    
    cv_list = cv_to_fold(cv,len(train.index))

    train = train.replace([np.inf, -np.inf], np.nan)
    train.fillna(0.0,inplace=True)
    train.to_csv("Degradation_6_with_smiles_feats.csv")

    # Do layer 1 outside of the chunking
    keep_f_all = ["IDENTIFIER","time"]
    keep_f_all.extend(copy.deepcopy(keep_f_withoutid))
    
    ms = str(int(time.time_ns()))

    hash_object = hashlib.sha1(ms.encode())
    hex_dig = hash_object.hexdigest()

    preds_own,mods_own = train_l1_func(train[keep_f_all],names=[hex_dig,hex_dig,hex_dig,hex_dig,hex_dig,hex_dig,hex_dig],adds=[n,n,n,n,n,n,n,n],cv=cv,outfile_modname=outfile_modname,n_jobs=num_jobs)

    preds_l1_train,skipped_train = apply_models(train.drop(["time","IDENTIFIER","system"],axis=1, errors='ignore')[keep_f_withoutid],known_rt=train["time"],row_identifiers=train["IDENTIFIER"],skip_cont=[hex_dig])
    preds_l1_train = pd.concat([preds_l1_train.reset_index(drop=True), preds_own], axis=1)

    test = p_infile
    test["IDENTIFIER"] = test["IDENTIFIER"].apply(replace_non_ascii)

    test[keep_f_withoutid] = scaler.transform(test[keep_f_withoutid])
    test.replace([np.inf, -np.inf], np.nan)
    test.fillna(0.0,inplace=True)

    print("Applying Layer 1...")

    preds_l1_test,skipped_test = apply_models(test.drop(["time","IDENTIFIER","system"],axis=1,errors='ignore')[keep_f_withoutid],row_identifiers=test["IDENTIFIER"],skip_cont=[],additional_models=mods_own)
    preds_l1_test.fillna(0.0,inplace=True)

    preds_diff_l1 = (preds_l1_test.loc[:,preds_l1_test.columns!="IDENTIFIER"]-preds_l1_test.loc[:,preds_l1_test.columns!="IDENTIFIER"].min())/(preds_l1_test.loc[:,preds_l1_test.columns!="IDENTIFIER"].max()-preds_l1_test.loc[:,preds_l1_test.columns!="IDENTIFIER"].min())
    preds_diff_l1.fillna(0.0,inplace=True)
    preds_diff_l1 = preds_diff_l1.loc[:,[c for c in preds_diff_l1.columns if c.endswith("xgb")]]
    dist_l1 = preds_diff_l1.apply(calc_overlap_compounds)

    dist_l1 = dist_l1.sort_values()
    plot_setups = dist_l1 #[:5]

    print("Applying Layer 2...")

    preds_l2_test,preds_l2_train = apply_l2(preds_l1_train,preds_l1_test,cv_list=cv_list,name=k)
    
    print("Applying Layer 3...")

    preds_l3_train,preds_l3_test,coefs = train_l3(preds_l2_train,preds_l2_test,cv=cv)

    outfilel3 = open("%s.csv" % (outfile),"w")
    outfilel3train = open("%s_train.csv" % (outfile),"w")
    
    preds_l3_train.columns = ["identifiers","predictions","tR"]
    preds_l3_test.columns = ["identifiers","predictions"]
    
    preds_l3_test.to_csv(outfilel3,index=False)
    preds_l3_train.to_csv(outfilel3train,index=False)
    outfilel3.close()
    outfilel3train.close()
    
    print("Done, predictions can be found here: %s.csv" % (outfile))
    print("===========")
    
    if len(outfile_modname) > 0:
        rem_files = ["mods_l1/%s_brr.pickle" % (k),
                    "mods_l1/%s_SVM.pickle" % (k),
                    "mods_l1/%s_xgb.pickle" % (k),
                    "mods_l1/%s_adaboost.pickle" % (k),
                    "mods_l1/%s_lasso.pickle" % (k)]
        for fn in rem_files:
            if os.path.exists(fn):
                os.remove(fn)
            else:
                print("Can not remove %s file. You need to remove it manually." % fn)

    return preds_l3_train, preds_l3_test, plot_setups, preds_l1_test, coefs, test

if __name__ == "__main__":
    make_preds(reference_infile="datasets/aicheler_data_features.csv",pred_infile="datasets/lm_features.csv")