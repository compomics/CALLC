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
from sklearn.preprocessing import StandardScaler

import pandas as pd

from trainl1 import train_l1_func

adds=["_r1"]
n_all = [100]

def remove_low_std(X,std_val=0.01):
	rem_f = []
	std_dist = X.std(axis=0)
	rem_f.extend(list(std_dist.index[std_dist<std_val]))
	return(rem_f)

def remove_high_cor(X,upp_cor=0.98,low_cor=-0.98):
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
	sets_dict = {}

	unique_systems = list(set(infile["system"]))
	
	for us in unique_systems:
		temp_us = infile[infile["system"]==us]
		sets_dict[us] = infile[infile["system"]==us]

	return(sets_dict)

def move_models(k):
	cmd = "mv mods_l1/%s*.pickle mods_l1/temp/" % (k)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
	p.communicate()

def remove_models(k,n):
	cmd = "rm -rf mods_l1/%s_*%s.pickle" % (k,n)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
	p.communicate()

def move_models_back(k):
	cmd = "mv mods_l1/temp/%s*.pickle mods_l1/" % (k)
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
	p.communicate()

def cv_to_fold(cv,num_ins):
	ret_vec = [0]*num_ins
	counter_f = 0
	for train,test in cv.split(ret_vec):
		for t in test:
			ret_vec[t] = counter_f
		counter_f += 1
	return(ret_vec)

def replace_non_ascii(ident):
	try:
		return "".join([str(s) for s in ident if s.isascii()])
	except TypeError:
		try:
			ident = str(ident)
			return "".join([str(s) for s in ident if s.isascii()])
		except TypeError:
			return "Non-ident"

def main(infilen="datasets/Degradation_6_with_smiles_feats.csv"):
	global adds
	global n_all

	infile = pd.read_csv(infilen)	
	infile.fillna(0.0,inplace=True)
	try:
		keep_f = [x.strip() for x in open("features/selected_features_v2.txt", encoding="utf-8").readlines()]
		print(list(infile.columns))
		keep_f.extend(["time","IDENTIFIER","system"])
		infile = infile.loc[:,keep_f]
		keep_f_features_only = [f for f in keep_f if f not in ["time","IDENTIFIER","system"]]
	except IOError:
		infile,keep_f = sel_features(infile)
		outfile = open("features/selected_features_v2.txt","w")
		outfile.write("\n".join(list(keep_f)))
		keep_f_features_only = [f for f in keep_f if f not in ["time","IDENTIFIER","system"]]

	scaler = StandardScaler()
	print(list(infile.columns))
	print(infile["IDENTIFIER"])
	infile["IDENTIFIER"] = infile["IDENTIFIER"].apply(replace_non_ascii)
	infile[keep_f_features_only] = scaler.fit_transform(infile[keep_f_features_only])
	#infile["IDENTIFIER"] = ["mol_"+str(i) for i in range(len(infile["IDENTIFIER"]))]

	sets = get_sets(infile)

	for k in sets.keys():
		print(k)
		selected_set = sets[k]
		select_index = range(len(selected_set.index))
		if len(select_index) < 3: continue

		n = 42

		train = selected_set
		
		if len(train.index) > 20: cv = KFold(n_splits=20)
		else: cv = KFold(n_splits=len(train.index))
		cv_list = cv_to_fold(cv,len(train.index))

		print("Training L1 %s,%s,%s" % (k,n,42))
		
		try:
			preds_own = train_l1_func(train,names=[k,k,k,k,k,k,k],adds=[n,n,n,n,n,n,n,n],cv=cv)
		except:
			continue

		print(train["time"])
		print(preds_own)

if __name__ == "__main__":
	main()
