from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.model_selection import GridSearchCV

from scipy.stats import randint
from scipy.stats import uniform

from numpy import arange

from sklearn.feature_selection import RFE

from scipy.stats import pearsonr

from operator import itemgetter
from numpy import median
from collections import Counter
    
def train_xgb(X,y,n_jobs=16,cv=None):
    preds = []
    featuresSE = [feat for feat in list(X) if "RtGAMSE" in feat]
    features = [feat.split("+")[0]+"+RtGAM" for feat in featuresSE]

    index = 0
    Xse = X[featuresSE]
    Xfe = X[features]

    selected_feat = [feat for feat in list(X) if feat.endswith("+RtGAM")]


    model = ElasticNet()
    crossv_mod = clone(model)
    ret_mod = clone(model)

    X = X[selected_feat]

    set_reg = [0.01,1.0,10.0,100.0,1000.0,10000.0,10000.0,100000.0,1000000.0,1000000000,1000000]
    set_reg.extend([x/2 for x in set_reg])
    set_reg.extend([x/3 for x in set_reg])
    
    params = {
       'alpha': set_reg,
       'l1_ratio' : [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
       'copy_X':[True],
       'normalize' : [False],
       'positive' : [True],
       'fit_intercept'  : [True,False]
    }

    grid = GridSearchCV(model, params,cv=cv,scoring='mean_absolute_error',verbose=0,n_jobs=n_jobs,refit=True)
    grid.fit(X,y)
    
    cv_pred = cv
    crossv_mod.set_params(**grid.best_params_)
    preds = cross_val_predict(crossv_mod, X=X, y=y, cv=cv_pred, n_jobs=n_jobs, verbose=0)

    ret_mod.set_params(**grid.best_params_)
    ret_mod.fit(X,y)

    coef_indexes = [i for i,coef in enumerate(ret_mod.coef_) if coef > 0.0]

    return(ret_mod,preds,selected_feat)

def train_l3(knowns,unknowns,cv=None):
    cols_known = list(knowns.columns)
    
    try: 
        unknowns = unknowns[cols_known]
    except: 
        cols_known.remove("time")
        unknowns = unknowns[cols_known]

    model,preds_train,selected_feat = train_xgb(knowns.drop(["time","IDENTIFIER"],axis=1, errors='ignore'),knowns["time"],cv=cv)
    preds_test = model.predict(unknowns.drop(["time","IDENTIFIER"],axis=1, errors='ignore')[selected_feat])

    knowns["preds"] = preds_train
    unknowns["preds"] = preds_test

    coef_indexes = [i for i,coef in enumerate(model.coef_) if coef > 0.0]
    coefs_list = [(name,model.coef_[i]) for i,name in enumerate(list(unknowns.drop(["time","IDENTIFIER"],axis=1, errors='ignore')[selected_feat])) if i in coef_indexes]
    
    coefs_tot = 0.0
    for mod_name,coef in coefs_list:
        if coef > 0.01:
            print("Layer 3 fitted coefficient (%s): %.3f" % (mod_name.rstrip("42+RtGAM"),coef))
            coefs_tot += coef
    print("Layer 3 sum coefficients: %.3f" % (coefs_tot))
    
    try: return(knowns[["IDENTIFIER","preds","time"]],unknowns[["IDENTIFIER","preds","time"]],coefs_list)
    except: return(knowns[["IDENTIFIER","preds","time"]],unknowns[["IDENTIFIER","preds"]],coefs_list)
