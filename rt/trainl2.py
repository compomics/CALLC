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
import pandas as pd

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

def apply_l2(known_all,unknown_all,name="cake",ignore_cols=["IDENTIFIER","time"],cv_list=None):
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

    print(known_all)
    input("stop")
    print(unknown_all)
    input("stop2")

    infile_known_handle = open("temp/tempKnownsl2.csv","w")
    infile_unknown_handle = open("temp/tempUnknownsl2.csv","w")
    infile_fold_handle = open("temp/tempFolds.txt","w")

    known_all.to_csv(infile_known_handle,index=False)
    unknown_all.to_csv(infile_unknown_handle,index=False)
    infile_fold_handle.write("\n".join(map(str,cv_list)))

    infile_known_handle.close()
    infile_unknown_handle.close()
    infile_fold_handle.close()

    preds,train_preds = call_ghostbusters()

    return(preds,train_preds)
