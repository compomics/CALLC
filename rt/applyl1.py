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

from os import listdir
from os.path import isfile, join
import pickle
import pandas
from sklearn.preprocessing import maxabs_scale
import xgboost as xgb
try:
    import copy_reg
except:
    import copyreg as copy_reg

import copyreg as copy_reg

def apply_models(X,outfile="",model_path="mods_l1/",known_rt=[],row_identifiers=[],skip_cont=[],additional_models=[]):
    """
    Apply the models from Layer 1
    
    Parameters
    ----------
    X : pd.DataFrame
        dataframe with molecular descriptors
    outfile : str
        specify the outfile
    mol_path : str
        path to models that need to be applied in Layer 1
    known_rt : list
        list with known retention times (equal to order in X)
    row_identifiers : list
        identifiers for each row (equal to order in X)
    skip_cont : list
        skip these models (provide file names)
    
    Returns
    -------
    list
        list with predictions
	list
		list with skipped models
    """
    model_fn = [join(model_path,f) for f in listdir(model_path) if isfile(join(model_path, f))]
 
    if len(additional_models) > 0:
        model_fn.extend(additional_models)

    preds = []
    t_preds = []
    skipped = []

    if len(row_identifiers) > 0: preds.append(list(row_identifiers))
    if len(known_rt) > 0: preds.append(list(known_rt))

    cnames = []
    if len(row_identifiers)  > 0: cnames.append("IDENTIFIER")
    if len(known_rt)  > 0: cnames.append("time")
    
    for f in model_fn:
        con = False
        for skip in skip_cont:
            compare = f.split("_")
            compare.pop()
            compare = "_".join(compare)
            if skip == compare:
                skipped.append(f.replace(".pickle",""))
                con = True
        if con: continue
        print("Applying model: %s" % (f))
        if f.endswith(".pickle"):
            with open(f,"rb") as model_f:
                try: model = pickle.load(model_f,encoding='latin1')
                except Exception as e:
                    print("Unable to load: %s" % (model_f))
                    print(e)
                    continue
        elif f.endswith(".json"):
            try:
                model = xgb.Booster()
                model.load_model(f)
            except:
                continue
        else:
            continue

        try: 
            temp_preds = model.predict(X)
        except Exception as e:
            try:
                del model.enable_categorical
                temp_preds = model.predict(X)
            except Exception as e2:
                print("Could not execute: %s" % (join(model_path, f)))
                print(e)
                continue
        preds.append(temp_preds)
        cnames.append(f.split("/")[-1].replace(".pickle","").replace(".json",""))

    preds = zip(*preds)
    preds = list(map(list,preds))

    print("Concatting predictions Layer 1...")
    preds = pandas.DataFrame(preds)
    preds.columns = cnames
    print("Returning predictions Layer 1...")
    return(preds,skipped)