import subprocess

from random import shuffle
from sklearn.model_selection import KFold

import pandas as pd

from trainl1 import train_l1_func
from applyl1 import apply_models
from trainl2 import apply_l2
from trainl3 import train_l3

import random

adds=["_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  "_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  "_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  "_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  "_r1","_r2","_r3","_r4","_r5",#,#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5"#,#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  #"_r1","_r2","_r3","_r4","_r5",#"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
	  ]#"_r1","_r2","_r3","_r4","_r5"]#,"_r6","_r7","_r8","_r9","_r10","_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20"]

			  #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",#"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
			  #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",#"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
			  #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",#"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20",
			  #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10"]#,"_r11","_r12","_r13","_r14","_r15","_r16","_r17","_r18","_r19","_r20"]
			  #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",
			  #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10",
			  #"_r1","_r2","_r3","_r4","_r5","_r6","_r7","_r8","_r9","_r10"]
n_all = [20,20,20,20,20,#20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
		 #30,30,30,30,30,#30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
		 40,40,40,40,40,#40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,
		 #50,50,50,50,50,#0,50,50,50,50,50,50,50,50,50,50,50,50,50,50,
		 60,60,60,60,60,#60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,
		 #70,70,70,70,70,#70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,
		 80,80 ,80 ,80 ,80, #,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80 ,80,
		 #90,90,90,90,90,#90,90,90,90,90,90,90,90,90,90,90,90,90,90,80,
		 100,100,100,100,100#,#100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
		 #110,110,110,110,110,#110,110,110,110,110,110,110,110,110,110,110,110,110,110,110,
		 #120,120,120,120,120#,#120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,
		 #130,130,130,130,130,#130,130,130,130,130,130,130,130,130,130,130,130,130,130,140,
		 #140,140,140,140,140,#40,140,140,140,140,140,140,140,140,140,140,140,140,140,140,
		 #150,150,150,150,150,#150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,
		 #160,160,160,160,160,#160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,
		 #170,170,170,170,170,#170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,
		 #180,180,180,180,180,#180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,
		 ]#190,190,190,190,190]#,190,190,190,190,190,190,190,190,190,190,190,190,190,190,190]

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
		if verbose: print "Removing the following features: %s" % rem_f
		
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
	for train,test in cv:
		for t in test:
			ret_vec[t] = counter_f
		counter_f += 1
	return(ret_vec)

def select_dupl(duplicated_indexes,unique_indexes,num_train=20,min_num_test=10):
	#make a hard copy?
	#print duplicated_indexes
	shuffle(duplicated_indexes)
	#print duplicated_indexes
	#raw_input()
	shuffle(unique_indexes)
	sel_train = []
	sel_test = []
	if len(duplicated_indexes) < num_train:
		sel_train.extend(duplicated_indexes)
		#tot en met?
		sel_train.extend(unique_indexes[0:num_train - len(duplicated_indexes)])
		sel_test.extend(unique_indexes[num_train - len(duplicated_indexes):])
	else:
		sel_train.extend(duplicated_indexes[0:num_train])
		sel_test.extend(unique_indexes)
		
	if len(sel_test) < min_num_test: return(False,False)
	return(sel_train,sel_test)

def get_unique_indexes(identifier,duplicates_identifiers):
	dupl = []
	uniq = []
	
	for c,i in enumerate(identifier):
		if i in duplicates_identifiers: dupl.append(c)
		else: uniq.append(c)
	return(dupl,uniq)

def main(infilen="train/retmetfeatures_new.csv"):
	global adds
	global n_all

	infile = pd.read_csv(infilen)	
	try:
		keep_f = [x.strip() for x in open("features/selected_features.txt").readlines()]
		infile = infile[keep_f]
	except IOError:
		infile,keep_f = sel_features(infile)
		outfile = open("features/selected_features.txt","w")
		outfile.write("\n".join(list(keep_f)))

	sets = get_sets(infile)

	#duplic_df = pd.read_csv("train/retmetfeatures_new.csv")
	#unique_df = pd.read_csv("train/retmetfeatures_new_nodup.csv")

	#all_id = list(duplic_df["IDENTIFIER"])
	#uniq = list(unique_df["IDENTIFIER"])
	#not_uniq = [u for u in all_id if u not in uniq]

	for k in sets.keys():
		#if k in ["","RIKEN","Taguchi_12","LIFE_old","Ales_18","PFR-TK72","Beck","Cao_HILIC","Eawag_XBridgeC18","FEM_lipids","FEM_long","FEM_orbitrap_plasma","FEM_orbitrap_urine","FEM_short","IPB_Halle","kohlbacher","Kojima","Krauss_21","Krauss","LIFE_new","LIFE_old","Mark","Matsuura_15","Matsuura","MPI_Symmetry","MTBLS20","MTBLS36","MTBLS38","MTBLS39","MTBLS87","Nikiforos","Otto","Stravs_22","Stravs","Taguchi","Takahashi","Tohge","Toshimitsu","UFZ_Phenomenex","UniToyama_Atlantis"]: continue
		if k != "Mark": continue
		#print k
		for ind in range(len(n_all)):
			selected_set = sets[k]
			select_index = range(len(selected_set.index))
			#if len(select_index) < 101: continue

			n = n_all[ind]

			if n > len(selected_set.index): continue

			#dupl_indexes,uniq_indexes = get_unique_indexes(selected_set["IDENTIFIER"],not_uniq)
			#sel_train,sel_test = select_dupl(dupl_indexes,uniq_indexes,num_train=n,min_num_test=10)



			shuffle(select_index)
			train = selected_set.iloc[select_index[0:n],] #
			test = selected_set.iloc[select_index[n:],] #
			
			if len(select_index[n:]) < 10: continue

			cv = KFold(len(train.index),n_folds=10)

			cv_list = cv_to_fold(cv,len(train.index))

			print "Training L1 %s,%s,%s" % (k,n,adds[ind])

			move_models(k)
			preds_own = train_l1_func(train,names=[k,k,k,k,k,k,k],adds=[n,n,n,n,n,n,n,n],cv=cv)

			print "Applying L1 %s,%s,%s" % (k,n,adds[ind])

			preds_l1_train,skipped_train = apply_models(train.drop(["time","IDENTIFIER","system"],axis=1),known_rt=train["time"],row_identifiers=train["IDENTIFIER"],skip_cont=[k])
			preds_l1_test,skipped_test = apply_models(test.drop(["time","IDENTIFIER","system"],axis=1),known_rt=test["time"],row_identifiers=test["IDENTIFIER"])			
		
			preds_l1_train = pd.concat([preds_l1_train.reset_index(drop=True), preds_own], axis=1)

			print "Applying L2 %s,%s,%s" % (k,n,adds[ind])

			preds_l2_test,preds_l2_train = apply_l2(preds_l1_train,preds_l1_test,cv_list=cv_list,name=k)

			#rem_col = preds_l1_train.drop(["time","IDENTIFIER"],axis=1).columns
			#rem_col = [r for r in rem_col if r in preds_l2_train.columns]
			#preds_l2_train = preds_l2_train.drop(rem_col,axis=1)
			#preds_l2_test = preds_l2_test.drop(rem_col,axis=1)

			preds_l2_train = pd.concat([preds_l2_train.reset_index(drop=True),train.drop(["IDENTIFIER","system","time"],axis=1).reset_index(drop=True)], axis=1)
			preds_l2_test = pd.concat([preds_l2_test.reset_index(drop=True),test.drop(["IDENTIFIER","system","time"],axis=1).reset_index(drop=True)], axis=1)

			#print preds_l2_train
			#raw_input()
			#print preds_l2_test
			#raw_input()

			print "Applying L3 %s,%s,%s" % (k,n,adds[ind])

			preds_l3_train,preds_l3_test,coefs_list = train_l3(preds_l2_train,preds_l2_test,cv=cv)

			outfilel1 = open("test_preds/%s_preds_l1_%s%s.csv" % (k,n,adds[ind]),"w")
			outfilel2 = open("test_preds/%s_preds_l2_%s%s.csv" % (k,n,adds[ind]),"w")
			outfilel3 = open("test_preds/%s_preds_l3_%s%s.csv" % (k,n,adds[ind]),"w")
			outfilel = open("test_preds/%s_preds_ALL_%s%s.csv" % (k,n,adds[ind]),"w")

			outfilel1train = open("test_preds/%s_preds_train_l1_%s%s.csv" % (k,n,adds[ind]),"w")
			outfilel2train = open("test_preds/%s_preds_train_l2_%s%s.csv" % (k,n,adds[ind]),"w")
			outfilel3train = open("test_preds/%s_preds_train_l3_%s%s.csv" % (k,n,adds[ind]),"w")
			outfileltrain = open("test_preds/%s_ALL_train_l3_%s%s.csv" % (k,n,adds[ind]),"w")

			outfilel3_coefs = open("test_preds/%s_preds_l3_%s%s_elasticnet_coefs.csv" % (k,n,adds[ind]),"w")

			for line in coefs_list:
				model,coefs = line
				outfilel3_coefs.write("%s_%s%s\t%s\t%s\n" % (k,n,adds[ind],model,str(coefs)))
			outfilel3_coefs.close()

			preds_l1_test.to_csv(outfilel1,index=False)
			preds_l2_test.to_csv(outfilel2,index=False)
			preds_l3_test.to_csv(outfilel3,index=False)

			all_test = pd.concat([preds_l1_test.reset_index(drop=True),preds_l2_test.reset_index(drop=True),preds_l3_test.reset_index(drop=True)],axis=1)
			all_test = all_test.T.drop_duplicates().T
			all_test.to_csv(outfilel,index=False)


			preds_l1_train.to_csv(outfilel1train,index=False)
			preds_l2_train.to_csv(outfilel2train,index=False)
			preds_l3_train.to_csv(outfilel3train,index=False)

			all_train = pd.concat([preds_l1_train.reset_index(drop=True),preds_l2_train.reset_index(drop=True),preds_l3_train.reset_index(drop=True)],axis=1)
			all_train = all_train.T.drop_duplicates().T
			all_train.to_csv(outfileltrain,index=False)
			
			outfilel1.close()
			outfilel2.close()
			outfilel3.close()
			outfilel.close()

			outfilel1train.close()
			outfilel2train.close()
			outfilel3train.close()
			outfileltrain.close()

			remove_models(k,n)
			move_models_back(k)


if __name__ == "__main__":
	random.seed(42)
	main()
