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

import subprocess

from random import shuffle
from sklearn.model_selection import KFold

import pandas as pd

from trainl1 import train_l1_func
from applyl1 import apply_models
from trainl2 import apply_l2
from trainl3 import train_l3

import numpy as np
from numpy import median

import os

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

def make_preds(reference_infile="train_set_lpp2.csv",pred_infile="lmfeatures.csv",k="MASSTRPLAN",outfile="",extra_pred_file="",outfile_modname="",num_jobs=4,GUI_obj=None,ch_size=100000):
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
    
    ref_infile = pd.read_csv(reference_infile)

    # Make sure we have the correct data types
    dict_dtypes = dict(ref_infile.select_dtypes(include=['int']).apply(pd.to_numeric,downcast="integer").dtypes)
    float_dtypes = dict(ref_infile.select_dtypes(include=['float']).apply(pd.to_numeric,downcast="float").dtypes)
    dict_dtypes.update(float_dtypes)

    # See if we need to chunk the predictions to be memory efficient
    tot_preds = sum(1 for row in open(pred_infile,"r"))/ch_size
    p_infile = pd.read_csv(pred_infile,dtype=dict_dtypes,chunksize=ch_size)

    counter_fold = 0

    keep_f = [x.strip() for x in open("features/selected_features.txt").readlines()]
    
    keep_f.remove("system")
    ref_infile = ref_infile[keep_f]

    keep_f.remove("time")    

    n = len(ref_infile)
    remove_models(k,n)
   
    print("===========")
    print("Total number of train molecules with tR: %s" % (n))

    # Make sure that for the training data we do not have infinite or nan
    train = ref_infile
    train = train.replace([np.inf, -np.inf], np.nan)
    train = train.fillna(0.0)

    # Define the folds to make predictions
    cv = KFold(n_splits=10,shuffle=True,random_state=42)
    cv = list(cv.split(train.index))
    
    cv_list = cv_to_fold(cv,len(train.index))

    # Do layer 1 outside of the chunking
    preds_own = train_l1_func(train,names=[k,k,k,k,k,k,k],adds=[n,n,n,n,n,n,n,n],cv=cv,outfile_modname=outfile_modname,n_jobs=num_jobs)
    preds_l1_train,skipped_train = apply_models(train.drop(["time","IDENTIFIER","system"],axis=1, errors='ignore'),known_rt=train["time"],row_identifiers=train["IDENTIFIER"],skip_cont=[k])
    preds_l1_train = pd.concat([preds_l1_train.reset_index(drop=True), preds_own], axis=1)

    for test in p_infile:
        counter_fold += 1
        print("----------")
        print("Read chunk (out of %s): %s" % (int(tot_preds)+1,counter_fold))
        test = test[keep_f]    
        test = test.replace([np.inf, -np.inf], np.nan)
        test = test.fillna(0.0)

        print("Applying Layer 1...")

        preds_l1_test,skipped_test = apply_models(test.drop(["time","IDENTIFIER","system"],axis=1,errors='ignore'),row_identifiers=test["IDENTIFIER"])

        print("Applying Layer 2...")

        preds_l2_test,preds_l2_train = apply_l2(preds_l1_train,preds_l1_test,cv_list=cv_list,name=k)

        rem_col = preds_l1_train.drop(["time","IDENTIFIER"],axis=1, errors='ignore').columns
        rem_col = [r for r in rem_col if r in preds_l2_train.columns]
        preds_l2_train = preds_l2_train.drop(rem_col,axis=1)
        preds_l2_test = preds_l2_test.drop(rem_col,axis=1)
        
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

if __name__ == "__main__":
    make_preds(reference_infile="datasets/aicheler_data_features.csv",pred_infile="datasets/lm_features.csv")