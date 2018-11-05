from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import maxabs_scale

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression

from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint
from scipy.stats import uniform

import pandas as pd

import pickle


def train_model_l1(X,y,params,model,scale=False,n_jobs=8,cv = None,n_params=20):
    if scale: X = maxabs_scale(X)
    else: X = X.apply(lambda x: pd.to_numeric(x, errors="coerce"))

    crossv_mod = clone(model)
    ret_mod = clone(model)

    grid = RandomizedSearchCV(model, params,cv=cv,scoring="mean_absolute_error",verbose=0,n_jobs=n_jobs,n_iter=n_params,refit=False)
    grid.fit(X,y)
    cv_pred = cv
    crossv_mod.set_params(**grid.best_params_)
    preds = cross_val_predict(crossv_mod, X=X, y=y, cv=cv_pred, n_jobs=n_jobs, verbose=0)

    ret_mod.set_params(**grid.best_params_)
    ret_mod.fit(X,y)

    return(ret_mod,preds)

def train_l1_func(sets,
                  names=["Cake.lie","Cake.lie1","Cake.lie2","Cake.lie3","Cake.lie4","Cake.lie5"],
                  adds=["","","","","","","",""],
                  cv = None,
                  n_params=20,
                  outfile_modname="",
                  n_jobs=8):

    ret_preds = []

    model = Lasso()

    params = {
        "alpha" : uniform(0.0,10.0),
        "copy_X" : [True],
        "fit_intercept" : [True],
        "normalize" : [False],
        "precompute" : [True,False],
        "max_iter" : [500]
    }
    
    print("Training Layer 1 LASSO")
    model,preds = train_model_l1(sets.drop(["time","IDENTIFIER","system"],axis=1, errors="ignore"),
                                             sets["time"],params,model,
                                             cv = cv,n_params=n_params,
                                             n_jobs=n_jobs)

    outfile = open("preds_l1/%s_lasso%s.txt" % (names[0],adds[0]),"w")
    for val in zip(list(sets["IDENTIFIER"]),list(sets["time"]),preds):
        outfile.write("%s,%s,%s\n" % val)
    outfile.close()

    with open("mods_l1/%s_lasso.pickle" % (names[0]), "wb") as f:
           pickle.dump(model, f)
    
    if len(outfile_modname) > 0:
        with open("%s_lasso.pickle" % (outfile_modname), "wb") as f:
           pickle.dump(model, f)
    
    ret_preds.append(preds)
    ###################################################################################################

    model = AdaBoostRegressor()

    params = {
        "n_estimators": randint(10,100), #,1000,3000], #
           "learning_rate": uniform(0.01,0.5)
    }

    print("Training Layer 1 AdaBoost")
    model,preds = train_model_l1(sets.drop(["time","IDENTIFIER","system"],axis=1, errors="ignore"),
                                             sets["time"],params,model,
                                             cv = cv,n_params=n_params,
                                             n_jobs=n_jobs)

    outfile = open("preds_l1/%s_adaboost%s.txt" % (names[1],adds[1]),"w")
    for val in zip(list(sets["IDENTIFIER"]),list(sets["time"]),preds):
        outfile.write("%s,%s,%s\n" % val)
    outfile.close()

    with open("mods_l1/%s_adaboost.pickle" % (names[1]), "wb") as f:
           pickle.dump(model, f)
           
    if len(outfile_modname) > 0:
        with open("%s_adaboost.pickle" % (outfile_modname), "wb") as f:
           pickle.dump(model, f)

    ret_preds.append(preds)

       ###################################################################################################


    model = xgb.XGBRegressor()

    params = {
        "n_estimators" : randint(20,100),
        "max_depth" : randint(1,12),
        "learning_rate" : uniform(0.01,0.25),
        "gamma" : uniform(0.0,10.0),
        "reg_alpha" : uniform(0.0,10.0),
        "reg_lambda" : uniform(0.0,10.0)
    }
    
    print("Training Layer 1 XGBoost")    
    model,preds = train_model_l1(sets.drop(["time","IDENTIFIER","system"],axis=1, errors="ignore"),
                                             sets["time"],params,model,
                                             n_jobs=1,cv = cv,n_params=n_params)

    outfile = open("preds_l1/%s_xgb%s.txt" % (names[2],adds[2]),"w")
    for val in zip(list(sets["IDENTIFIER"]),list(sets["time"]),preds):
        outfile.write("%s,%s,%s\n" % val)
    outfile.close()

    with open("mods_l1/%s_xgb.pickle" % (names[2]), "wb") as f:
           pickle.dump(model, f)
           
    if len(outfile_modname) > 0:
        with open("%s_xgb.pickle" % (outfile_modname), "wb") as f:
           pickle.dump(model, f)

    ret_preds.append(preds)

              ###################################################################################################

    model = SVR()

    params = {
       "epsilon": uniform(0.01,100.0),
       "kernel": ["linear","rbf"],
       "degree": randint(1,12),
       "gamma" : uniform(0.000001,100),
       #"tol" : [1e-10]#,
       #"max_iter" : [2000000000]
    }
    
    print("Training Layer 1 SVR")
    model,preds = train_model_l1(sets.drop(["time","IDENTIFIER","system"],axis=1, errors="ignore"),
                                             sets["time"],params,model,
                                             scale=True,cv = cv,n_params=n_params,
                                             n_jobs=n_jobs)

    outfile = open("preds_l1/%s_SVM%s.txt" % (names[3],adds[3]),"w")
    for val in zip(list(sets["IDENTIFIER"]),list(sets["time"]),preds):
        outfile.write("%s,%s,%s\n" % val)
    outfile.close()

    with open("mods_l1/%s_SVM.pickle" % (names[3]), "wb") as f: 
           pickle.dump(model, f)
           
    if len(outfile_modname) > 0:
        with open("%s_SVM.pickle" % (outfile_modname), "wb") as f:
           pickle.dump(model, f)

    ret_preds.append(preds)

       ###################################################################################################

    model = ARDRegression()

    params = {
        "n_iter" : randint(100,1500),
        "alpha_1" : uniform(1e-10,1e-2),
        "lambda_1" : uniform(1e-10,1e-2),
        "threshold_lambda" : randint(1,10000),
    }
    
    print("Training Layer 1 BRR")
    model,preds = train_model_l1(sets.drop(["time","IDENTIFIER","system"],axis=1, errors="ignore"),
                                             sets["time"],params,model,
                                             cv = cv,n_params=n_params,
                                             n_jobs=n_jobs)

    outfile = open("preds_l1/%s_bayesianregr%s.txt" % (names[6],adds[6]),"w")
    for val in zip(list(sets["IDENTIFIER"]),list(sets["time"]),preds):
        outfile.write("%s,%s,%s\n" % val)
    outfile.close()


    with open("mods_l1/%s_brr.pickle" % (names[6]), "wb") as f:
           pickle.dump(model, f)

    if len(outfile_modname) > 0:
        with open("%s_brr.pickle" % (outfile_modname), "wb") as f:
           pickle.dump(model, f)
           
    ret_preds.append(preds)

       ###################################################################################################


    ret_preds = pd.DataFrame(ret_preds).transpose()


    ret_preds.columns = ["%s_lasso" % (names[0]),
                         "%s_adaboost"  % (names[1]),
                         "%s_xgb" % (names[2]),
                         "%s_SVM" % (names[3]),
                         "%s_brr"  % (names[4])]

    return(ret_preds)


    
