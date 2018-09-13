import subprocess
import pandas as pd

def call_ghostbusters(infile_known="temp/tempKnownsl2.csv",infile_unknown="temp/tempUnknownsl2.csv",fold_list="temp/tempFolds.txt"): #df_known,df_unknown,
    cmd = "Rscript makeGAM.R %s %s %s" % (infile_known,infile_unknown,fold_list)
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    out, err = p.communicate()
    
    preds = pd.read_csv("GAMpredTemp.csv")
    train_preds = pd.read_csv("GAMtrainTemp.csv")

    return(preds,train_preds)

def apply_l2(known_all,unknown_all,name="cake",ignore_cols=["IDENTIFIER","time"],cv_list=None):
    ret_preds = []
    ret_preds_train = []
    cnames = []
    
    known_all.index = known_all["IDENTIFIER"]
    unknown_all.index = unknown_all["IDENTIFIER"]

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
