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
from sklearn.preprocessing import StandardScaler

from trainl1 import train_l1_func
from applyl1 import apply_models
#from trainl2 import apply_l2
from trainl2_pygam import apply_l2
from trainl3 import train_l3

import random

from pickle import dump
from pickle import load
import numpy as np

adds=["_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      "_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      "_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      "_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      "_r1","_r2","_r3","_r4","_r5",#,#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5"#,#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
      ]#"_r1","_r2","_r3","_r4","_r5"]#,"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20"]

              #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",#"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
              #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",#"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
              #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",#"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
              #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10"]#,"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20"]
              #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",
              #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",
              #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10"]
n_all = [20,20,20,20,20,#20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
         #30,30,30,30,30,#30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
         40,40,40,40,40,#40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,
         #50,50,50,50,50,#0,50,50,50,50,50,50,50,50,50,50,50,50,50,50,
         60,60,60,60,60,#60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,
         #70,70,70,70,70,#70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,
         80,80 ,80 ,80 ,80, #,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80,
         #90,90,90,90,90,#90,90,90,90,90,90,90,90,90,90,90,90,90,90,80,
         100,100,100,100,100#,#100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
         #110,110,110,110,110,#110,110,110,110,110,110,110,110,110,110,110,110,110,110,110,
         #120,120,120,120,120#,#120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,
         #130,130,130,130,130,#130,130,130,130,130,130,130,130,130,130,130,130,130,130,140,
         #140,140,140,140,140,#40,140,140,140,140,140,140,140,140,140,140,140,140,140,140,
         #150,150,150,150,150,#150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,
         #160,160,160,160,160,#160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,
         #170,170,170,170,170,#170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,
         #180,180,180,180,180,#180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,
         ]#190,190,190,190,190]#,190,190,190,190,190,190,190,190,190,190,190,190,190,190,190]

def remove_low_std(X,std_val=0.01):
    """
    Remove features below this standard deviation
    
    Parameters
    ----------
    X : pd.DataFrame
        analytes with their features (i.e. molecular descriptors)
    std_val : float
        float value to cut-off removing features

    Returns
    -------
    list
        list with features to remove
    """
    rem_f = []
    std_dist = X.std(axis=0)
    rem_f.extend(list(std_dist.index[std_dist<std_val]))
    return(rem_f)

def remove_high_cor(X,upp_cor=0.98,low_cor=-0.98):
    """
    Remove features that correlate too high
    
    Parameters
    ----------
    X : pd.DataFrame
        analytes with their features (i.e. molecular descriptors)
    upp_cor : float
        float value to upper cut-off removing features
    low_cor : float
        float value to lower cut-off removing features

    Returns
    -------
    list
        list with features to remove
    """
    rem_f = []
    keep_f = []

    new_m = X.corr()
    new_m = list(new_m.values)
    for i in range(len(new_m)):
        for j in range(len(new_m[i])):
            if i == j: continue
            if new_m[i][j] > upp_cor or new_m[i][j] < low_cor:
                if X.columns[j] not in keep_f:
                    rem_f.append(X.columns[j])
                    keep_f.append(X.columns[i])
    return(rem_f)

def sel_features(infile,verbose=True,remove_std=True,remove_cor=True,std_val=0.01,upp_cor=0.99,low_cor=-0.99,ignore_cols=["system","IDENTIFIER","time"]):
    """
    Remove features that correlate too high
    
    Parameters
    ----------
    infile : pd.DataFrame
        analytes with their features (i.e. molecular descriptors)
    verbose : boolean
        print messages
    remove_std : boolean
        flag to remove low std features
    remove_cor : boolean
        flag to remove high cor features
    std_val : float
        float value to cut-off removing features
    upp_cor : float
        float value to upper cut-off removing features
    low_cor : float
        float value to lower cut-off removing features
    ignore_cols : list
        ignore these features

    Returns
    -------
    pd.DataFrame
        dataframe without the removed features
    pd.Series
        vector with features still part of the dataframe
    """
    if remove_std or remove_cor:
        rem_f = []

        if remove_std: rem_f.extend(remove_low_std(infile,std_val=std_val))
        if remove_cor: rem_f.extend(remove_high_cor(infile,))

        rem_f = list(set(rem_f))
        [rem_f.remove(x) for x in rem_f if x in ignore_cols]
        if verbose: print("Removing the following features: %s" % rem_f)
        
        infile.drop(rem_f, axis=1, inplace=True)
    return(infile,infile.columns)

def get_sets(infile):
    """
    Remove features that correlate too high
    
    Parameters
    ----------
    infile : pd.DataFrame
        analytes with their features (i.e. molecular descriptors)

    Returns
    -------
    dict
        sliced up dataframe per dataset
    """
    sets_dict = {}

    unique_systems = list(set(infile["system"]))
    
    for us in unique_systems:
        temp_us = infile[infile["system"]==us]
        sets_dict[us] = infile[infile["system"]==us]

    return(sets_dict)

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
    cmd = "rm -rf mods_l1/%s_*%s.pickle" % (k,n)
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

def main(infilen="retmet_features_streamlit.csv"):
    """
    Make predictions for the evaluation of CALLC
    
    Parameters
    ----------
    infilen : str
        location of train data

    Returns
    -------

    """
    global adds
    global n_all

    infile = pd.read_csv(infilen)
    infile.fillna(0.0,inplace=True)    
    sets = get_sets(infile)
    
    try:
        keep_f = [x.strip() for x in open("features/selected_features_v2.txt", encoding="utf-8").readlines()]
        #infile = infile[keep_f]
        keep_f_features_only = [f for f in keep_f if f not in ["time","IDENTIFIER","system"]]
        infile[keep_f_features_only] = infile[keep_f_features_only].applymap(lambda x: np.nan if isinstance(x, str) else x)
        infile.fillna(0.0,inplace=True)
    except IOError:
        infile,keep_f = sel_features(infile)
        outfile = open("features/selected_features_v2.txt","w")
        outfile.write("\n".join(list(keep_f)))
        
        keep_f_features_only = [f for f in keep_f if f not in ["time","IDENTIFIER","system"]]        
        infile[keep_f_features_only] = infile[keep_f_features_only].applymap(lambda x: np.nan if isinstance(x, str) else x)
        infile.fillna(0.0,inplace=True)
    
    keep_f_time_sys_ident = list(keep_f)
    keep_f_time_sys_ident.extend(["time","IDENTIFIER","system"])

    scaler = load(open('scaler.pkl', 'rb'))


    infile[keep_f_features_only] = scaler.transform(infile[keep_f_features_only])

    infile.fillna(0.0,inplace=True)
    infile.replace(np.inf, 0.0, inplace=True)
    infile.replace(-np.inf, 0.0, inplace=True)

    sets = get_sets(infile)

    for k in sets.keys():
        #if k in ["Waters ACQUITY UPLC with Synapt G1 Q-TOF","Ales_18",
        #         "CS5","CS4","FEM_long","IJM_TEST","Matsuura_15","Krauss","CS17","ABC",
        #         "SNU_RP_108","CS11","CS20","SNU_RP_indole_order","CS12","CS23","MTBLS87",
        #         "Takahashi","Stravs","UniToyama_Atlantis","CS21","SNU_RP_30","RP_PSB_HF",
        #         "cecum_JS","CS14","MTBLS17","MTBLS38","Krauss_21","HILIC_BDD_2","LIFE_new",
        #         "FEM_orbitrap_plasma","Ken","CS15","SNU_RP_10","Janssen","cuw","CS3","Matsuura",
        #         "MTBLS52","BDD_C18","JKD_Probiotics","CS16","CS7","KI_GIAR_zic_HILIC_pH2_7",
        #         "RPFDAMM","HIILIC_tip","FEM_lipids","FEM_orbitrap_urine",
        #         "OBSF","CS19","RIKEN","CS10","MTBLS39","PFR-TK72","Cao_HILIC","CS8",
        #         "IPB_Halle","CS13","Tohge","RPMMFDA","UFZ_Phenomenex","Kojima","Nikiforos",
        #         "Toshimitsu","Stravs_22","CS22","Huntscha","Mark","SMRT","SNU-test","MTBLS4",
        #         "MTBLS36","SNU_RP_70","CS9"]:
        #    continue
        if k == "Waters ACQUITY UPLC with Synapt G1 Q-TOF":
            continue
        if k not in ["MPI_Symmetry","PFR-TK72","Cao_HILIC","Eawag_Xbridge","UniToyama_Atlantis","LIFE_old","MTBLS4","RIKEN","MTBLS52","Beck","FEM_lipids","Nikiforos","MTBLS36","FEM_short","MTBLS","LIFE_new","MTBLS20","FEM_orbitrap_urine","Matsuura_15","Kojima","MTBLS87","MTBLS38","Huntscha","Aicheler","Matsuura","Takahashi","Ken","FEM_orbitrap_plasma","UFZ_phenomenex","Otto","Tohge","MTBLS19","FEM_long","Ales_18","Taguchi","IPB_Halle","Stravs_22","Krauss","MTBLS39"]:
            continue
        selected_set = sets[k]

        kf = KFold(shuffle=True,random_state=1,n_splits=10)

        if len(selected_set.index) < 20: continue

        exp_counter = 0
        ind = -1
        for train_index, test_index in kf.split(selected_set):
            ind += 1
            exp_counter += 1
            n = exp_counter
            print("TRAIN:", train_index, "TEST:", test_index)
            print(selected_set)


            train = selected_set.iloc[train_index]
            test = selected_set.iloc[test_index]

            cv = KFold(n_splits=10,shuffle=True,random_state=42)
            cv = list(cv.split(train.index))
                
            cv_list = cv_to_fold(cv,len(train.index))    

            print("Training L1 %s,%s,%s" % (k,n,adds[ind]))

            move_models(k)
            preds_own, mods_own = train_l1_func(train[keep_f_time_sys_ident],names=[k,k,k,k,k,k,k],adds=[n,n,n,n,n,n,n,n],cv=cv)

            print("Applying L1 %s,%s,%s" % (k,n,adds[ind]))

            preds_l1_train,skipped_train = apply_models(train.drop(["time","IDENTIFIER","system"],axis=1)[keep_f],known_rt=train["time"],row_identifiers=train["IDENTIFIER"],skip_cont=[k])

            preds_l1_test,skipped_test = apply_models(test.drop(["time","IDENTIFIER","system"],axis=1)[keep_f],known_rt=test["time"],row_identifiers=test["IDENTIFIER"],additional_models=mods_own) 
        
            preds_l1_train = pd.concat([preds_l1_train.reset_index(drop=True), preds_own], axis=1)

            print("Applying L2 %s,%s,%s" % (k,n,adds[ind]))

            preds_l2_test,preds_l2_train = apply_l2(preds_l1_train,preds_l1_test,cv_list=cv_list,name=k)

            preds_l2_train = pd.concat([preds_l2_train.reset_index(drop=True),train.drop(["IDENTIFIER","system","time"],axis=1).reset_index(drop=True)], axis=1)
            preds_l2_test = pd.concat([preds_l2_test.reset_index(drop=True),test.drop(["IDENTIFIER","system","time"],axis=1).reset_index(drop=True)], axis=1)

            preds_l2_test.drop(keep_f_features_only, axis=1, inplace=True)
            preds_l2_train.drop(keep_f_features_only, axis=1, inplace=True)

            print("Applying L3 %s,%s,%s" % (k,n,adds[ind]))

            preds_l3_train,preds_l3_test,coefs_list = train_l3(preds_l2_train,preds_l2_test,cv=cv)

            outfilel1 = open("test_preds/%s_preds_l1_%s%s.csv" % (k,n,adds[ind]),"w")
            outfilel2 = open("test_preds/%s_preds_l2_%s%s.csv" % (k,n,adds[ind]),"w")
            outfilel3 = open("test_preds/%s_preds_l3_%s%s.csv" % (k,n,adds[ind]),"w")
            outfilel = open("test_preds/%s_preds_ALL_%s%s.csv" % (k,n,adds[ind]),"w")

            outfilel1train = open("test_preds/%s_preds_train_l1_%s%s.csv" % (k,n,adds[ind]),"w")
            outfilel2train = open("test_preds/%s_preds_train_l2_%s%s.csv" % (k,n,adds[ind]),"w")
            outfilel3train = open("test_preds/%s_preds_train_l3_%s%s.csv" % (k,n,adds[ind]),"w")
            outfileltrain = open("test_preds/%s_ALL_train_l3_%s%s.csv" % (k,n,adds[ind]),"w")

            outfilel3_coefs = open("test_preds/%s_preds_l3_%s%s_elasticnet_coefs.csv" % (k,n,adds[ind]),"w")

            for line in coefs_list:
                model,coefs = line
                outfilel3_coefs.write("%s_%s%s\t%s\t%s\n" % (k,n,adds[ind],model,str(coefs)))
            outfilel3_coefs.close()

            preds_l1_test.to_csv(outfilel1,index=False)
            preds_l2_test.to_csv(outfilel2,index=False)
            preds_l3_test.to_csv(outfilel3,index=False)

            all_test = pd.concat([preds_l1_test.reset_index(drop=True),preds_l2_test.reset_index(drop=True),preds_l3_test.reset_index(drop=True)],axis=1)
            all_test = all_test.T.drop_duplicates().T
            all_test.to_csv(outfilel,index=False)


            preds_l1_train.to_csv(outfilel1train,index=False)
            preds_l2_train.to_csv(outfilel2train,index=False)
            preds_l3_train.to_csv(outfilel3train,index=False)

            all_train = pd.concat([preds_l1_train.reset_index(drop=True),preds_l2_train.reset_index(drop=True),preds_l3_train.reset_index(drop=True)],axis=1)
            all_train = all_train.T.drop_duplicates().T
            all_train.to_csv(outfileltrain,index=False)
            
            outfilel1.close()
            outfilel2.close()
            outfilel3.close()
            outfilel.close()

            outfilel1train.close()
            outfilel2train.close()
            outfilel3train.close()
            outfileltrain.close()

            remove_models(k,n)
            move_models_back(k)


if __name__ == "__main__":
    random.seed(42)
    main()
