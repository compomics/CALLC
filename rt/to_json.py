import os
import pickle
import xgboost as xgb

path = "mods_l1/"
files = os.listdir(path)

for f in files:
    if "_xgb" not in f:
        continue
    #if "Component_" not in f and "Degradation_" not in f:
    #    continue
    if not f.endswith(".json"):
        continue

    print(f)

    model = xgb.Booster()
    try:
        model.load_model(os.path.join(path,f))
    except Exception as e:
        print(e)
        print("-------")
        continue

    #mod = pickle.load(open(os.path.join(path,f),"rb"),encoding='latin1')
    #mod_name = f.split(".")[0]
    #mod.save_model(f"{mod_name}.json")
