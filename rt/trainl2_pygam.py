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

from pygam import LinearGAM, s
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

def call_ghostbusters(infile_known="temp/tempKnownsl2.csv",infile_unknown="temp/tempUnknownsl2.csv",fold_list="temp/tempFolds.txt"): #df_known,df_unknown,
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
    cmd = "Rscript makeGAM.R %s %s %s" % (infile_known,infile_unknown,fold_list)
    print("Executing: ",cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    out, err = p.communicate()
    
    preds = pd.read_csv("GAMpredTemp.csv")
    train_preds = pd.read_csv("GAMtrainTemp.csv")

    return(preds,train_preds)

def apply_l2(known_all,unknown_all,name="cake",ignore_cols=["IDENTIFIER","time"],cv_list=None,top_cor=5):
    """
    Get the dataframe associated with this analysis
    
    Parameters
    ----------
    known_all : pd.DataFrame
        dataframe with known retention time, for Layer 2
	unknown_all : pd.DataFrame
		dataframe with unknown retention time, for Layer 2
    ignore_cols : list
        ignore these columns
    cv_list : list
        the folds to be used in Layer 2
    
    Returns
    -------
    pd.DataFrame
        test predictions
	pd.DataFrame
		train predictions
    """
    ret_preds = []
    ret_preds_train = []
    cnames = []
    
    known_all.index = known_all["IDENTIFIER"]
    unknown_all.index = unknown_all["IDENTIFIER"]

    df_return_train = pd.DataFrame(np.zeros((len(known_all.index), len(known_all.columns))))
    df_return_train.columns = known_all.columns
    df_return_train.index = known_all.index
    df_return_train["IDENTIFIER"] = known_all["IDENTIFIER"]
    df_return_train["time"] = known_all["time"]

    df_return_test = pd.DataFrame(np.zeros((len(unknown_all.index), len(unknown_all.columns))))
    df_return_test.columns = unknown_all.columns
    df_return_test.index = unknown_all.index
    df_return_test["IDENTIFIER"] = unknown_all["IDENTIFIER"]

    all_cor = []
    for c in known_all.columns:
        try:
            if c in ["IDENTIFIER","time"]:
                continue
            try:
                cor = spearmanr(known_all[c], known_all["time"])[0]
            except:
                continue
            all_cor.append(cor)
        except:
            continue
    min_cor = sorted(all_cor)[top_cor*-1]

    for c in known_all.columns:
        try:
            if c in ["IDENTIFIER","time"]:
                continue
            try:
                cor = spearmanr(known_all[c], known_all["time"])[0]
            except:
                continue

            if abs(cor) < min_cor:
                continue

            if cor < 0.0:
                constr = "monotonic_dec"
            else:
                constr = "monotonic_inc"

            unique_cv = list(set(cv_list))

            for cv_num in unique_cv:
                selection_instances = [True if fold_num != cv_num else False for fold_num in cv_list]
                selection_instances_test = [False if fold_num != cv_num else True for fold_num in cv_list]

                X_train = known_all.loc[selection_instances,c]
                y_train = known_all.loc[selection_instances,"time"]

                X_test = known_all.loc[selection_instances_test,c]
                y_test = known_all.loc[selection_instances_test,"time"]

                gam_model_cv = LinearGAM(s(0, constraints=constr, n_splines=10), verbose=True).fit(X_train, y_train)
                df_return_train.loc[selection_instances_test,c] = list(gam_model_cv.predict(X_test))
            
            print("--------------------------------")
            gam_model = LinearGAM(s(0, constraints=constr, n_splines=10), verbose=True).fit(known_all[c], known_all["time"])
            df_return_test.loc[:,c] = list(gam_model.predict(unknown_all[c]))
        except KeyError:
            continue
    non_feature_cols = ["IDENTIFIER","time"]
    
    df_return_test.columns = [c+"+RtGAM" if c not in non_feature_cols else c for c in df_return_test.columns]
    df_return_train.columns = [c+"+RtGAM" if c not in non_feature_cols else c for c in df_return_train.columns]
    
    return(df_return_test,df_return_train)
