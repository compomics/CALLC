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

def apply_models(X,outfile="",model_path="mods_l1/",known_rt=[],row_identifiers=[],skip_cont=[]):
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
    model_fn = [f for f in listdir(model_path) if isfile(join(model_path, f))]
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
        print("Applying model: %s" % (join(model_path, f)))
        with open(join(model_path, f),"rb") as model_f:
            try: model = pickle.load(model_f,encoding='latin1')
            except: print("Unable to load: %s" % (model_f))
        if "_SVM" in f:
            X_temp = maxabs_scale(X)
            preds.append(model.predict(X_temp))
            cnames.append(f.replace(".pickle",""))
            continue
        try: 
            temp_preds = model.predict(X)
        except:
            print("Could not execute: %s" % (join(model_path, f)))
            continue
        preds.append(temp_preds)
        cnames.append(f.replace(".pickle",""))

    preds = zip(*preds)
    preds = list(map(list,preds))

    print("Concatting predictions Layer 1...")
    preds = pandas.DataFrame(preds)
    preds.columns = cnames
    print("Returning predictions Layer 1...")
    return(preds,skipped)