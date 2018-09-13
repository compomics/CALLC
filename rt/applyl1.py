from os import listdir
from os.path import isfile, join
import pickle
import pandas
from sklearn.preprocessing import maxabs_scale

def apply_models(X,outfile="",model_path="mods_l1/",known_rt=[],row_identifiers=[],skip_cont=[]):
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
            except: continue #print("Unable to load: %s" % (model_f))
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