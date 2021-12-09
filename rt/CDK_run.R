
if (!require("rJava")) {
  install.packages("rJava", dependencies = TRUE,repos = "http://cran.us.r-project.org")
  library(rJava)
}

if (!require("rcdk")) {
  install.packages("rcdk", dependencies = TRUE,repos = "http://cran.us.r-project.org")
  library(rcdk)
}

if (!require("doParallel")) {
  install.packages("doParallel", repos="http://R-Forge.R-project.org")
  library(doParallel)
}

if (!require("doMC")) {
  install.packages("doMC", dependencies = TRUE, repos="http://R-Forge.R-project.org")
  library(doMC)
}


#setup parallel backend to use many processors
cores=detectCores()
cl <- makeCluster(1) #cores[1]-1) #not to overload your computer
registerDoParallel(cl)


#setwd("C:/Users/compo/Desktop/CCS")
setwd("./")

print(paste0("Converting SMILES..."))

myArgs <- commandArgs(trailingOnly = TRUE)

files = as.character(myArgs)

infile = read.csv(files[1],header=T) #read.csv("temp_smiles.csv",header=T) #
outfile_name = files[2] #"cdk_features.csv" #

#SMILES <- infile["SMILES"]
#snames <- infile["ident"]

#x <- data.frame(SMILES,snames)

x <- infile

print(paste0("Here 2..."))

all_smi <- c()
for (i in 1:nrow(x)) {
  smi <- rcdk::parse.smiles(as.character(unlist(x[i,"SMILES"]))) [[1]]
  smi1 <- rcdk::generate.2d.coordinates(smi)
  smi1 <- rcdk::get.smiles(smi,smiles.flavors(c('CxSmiles')))
  all_smi <- c(all_smi,smi1)
  #x$SMILES[i] <- smi1
  print(paste0(i," of ",nrow(x)))
}
x["SMILES"] <- all_smi

# select all possible descriptors
descNames <- rcdk::get.desc.names(type = "all")
# select only one descriptors. This helps to remove compounds that makes errors
descNames1 <- c('org.openscience.cdk.qsar.descriptors.molecular.BCUTDescriptor')

print(paste0("Checking for compound errors..."))


# calculate only 1 descriptor for all the molecules
mols_x <- rcdk::parse.smiles(as.character(unlist(x[1,"SMILES"])))
print(paste0("Here 5..."))
descs1_x <- rcdk::eval.desc(mols_x, descNames1)

print(paste0("Here 4..."))
for (i in 2:nrow(x)) {
  mols1 <- rcdk::parse.smiles(as.character(unlist(x[i,"SMILES"])))
  descs1_x[i,] <- rcdk::eval.desc(mols1, descNames1)
  print(paste0(i," of ",nrow(x)))
}
print(paste0("Here 5..."))
# remove molecules that have NA values with only one descriptor
x_na <- data.frame(descs1_x,x)

x_na_rem <- x_na #[stats::complete.cases(x_na), ]

x_na_rem <- x_na_rem [,-c(1:6)]

# computing the whole descriptos on the good on the clean dataset
print(paste0("Computing Chemical Descriptors 1 of ",nrow(x_na_rem)," ... Please wait"))
print(paste0(as.character(unlist(x_na_rem[1,"SMILES"]))))


mols_x1 <- rcdk::parse.smiles(as.character(unlist(x_na_rem[1,"SMILES"])))[[1]]

print(mols_x1)
descNames = descNames[descNames != "org.openscience.cdk.qsar.descriptors.molecular.LongestAliphaticChainDescriptor"]

rcdk::convert.implicit.to.explicit(mols_x1)
descs_x_loop <- rcdk::eval.desc(mols_x1, descNames)

get_desc <- function(i,x_na_rem){
  print(x_na_rem[i,"SMILES"])
  mols <- rcdk::parse.smiles(as.character(unlist(x_na_rem[i,"SMILES"])))[[1]]
  print(mols)
  rcdk::convert.implicit.to.explicit(mols)
  print(paste0(i," of ",nrow(x_na_rem)))
  return(data.frame(rcdk::eval.desc(mols, descNames)))
}



#for (i in 1:nrow(x_na_rem)) {
#  if (i != 2){
#    print(paste0(i," of ",nrow(x_na_rem)))
#    print(as.character(unlist(x_na_rem[i,"SMILES"])))
#    mols <- rcdk::parse.smiles(as.character(unlist(x_na_rem[i,"SMILES"])))[[1]]
#    rcdk::convert.implicit.to.explicit(mols)
#    print(paste0(i," of ",nrow(x_na_rem)))
#    print(mols)
#    print(rcdk::eval.desc(mols, descNames))
#    print(data.frame(rcdk::eval.desc(mols, descNames)))
#  }
#}

df_list <- foreach (i=1:nrow(x_na_rem)) %dopar% get_desc(i,x_na_rem)
datadesc <- do.call(rbind,df_list)

#for (i in 2:nrow(x_na_rem)) {
#  mols <- rcdk::parse.smiles(as.character(unlist(x_na_rem[i,"SMILES"])))[[1]]
#  rcdk::convert.implicit.to.explicit(mols)
#  descs_x_loop[i,] <- rcdk::eval.desc(mols, descNames)
#  print(paste0(i," of ",nrow(x_na_rem)))
#}
#datadesc <- data.frame(x_na_rem,descs_x_loop)

datadesc[is.na(datadesc)] <- 0
datadesc["ident"] <- infile["ident"]

write.csv(datadesc,outfile_name,quote=F,row.names=F)
