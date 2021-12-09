if (!require("mgcv")) {
  install.packages("mgcv", dependencies = TRUE, repos = "http://cran.us.r-project.org")
  library(mgcv)
}

#if (!require("doMC")) {
#  install.packages("doMC", dependencies = TRUE, repos="http://R-Forge.R-project.org")
#  library(doMC)
#}

if (!require("doParallel")) {
  install.packages("doParallel", dependencies = TRUE, repos = "http://cran.us.r-project.org")
  library(doParallel)
}

set.seed(42)

c1 <- makeCluster(2)
registerDoParallel(c1)

myArgs <- commandArgs(trailingOnly = TRUE)

files = as.character(myArgs)

knowns = read.csv(files[1],header=T)
unknowns = read.csv(files[2],header=T)
fold_list = read.csv(files[3],header=F)[,1]

par_gam <- function(i,c,x_t,y_t,knowns,pred_knowns,pred_knowns_se){
  library(mgcv)
  b_t <- try(gam(y_t~s(x_t,k=-2),method="ML",select=T,parallel=T))
  if (class(b_t) == "try-error") {
    return(rep(0,length(knowns[fold_list==i,c]))) #
    pred_knowns[fold_list==i] <- 0
    pred_knowns_se[fold_list==i] <- 100000}
  else {
    return(list(predict(b_t,data.frame(x_t=knowns[fold_list==i,c]), se.fit = FALSE,type="response"))) #$fit
                #,predict(b_t,data.frame(x_t=knowns[fold_list==i,c]), se.fit = TRUE,type="response")$se.fit)) 
  }
  
}

all_exp <- colnames(knowns)

for (c in colnames(knowns)) {
  if (c %in% c("IDENTIFIER","time")){ next }
  if (!(c %in% colnames(unknowns))) { next }
  
  x <- knowns[,c]
  y <- knowns[,"time"]
  
  b <- try(gam(y~s(x,k=-2),method="ML",select=T,parallel=T))

  if (class(b) == "try-error") { b <- try(gam(y~s(x,k=-2),method="ML",select=T,parallel=T))} #,k=-10
  if (class(b) == "try-error") {   
    unknowns[paste(c,"+","RtGAM",sep="")] <- 0
    knowns[paste(c,"+","RtGAM",sep="")] <- 0
    next}
   
  pred_knowns <- rep.int(0,length(x))
  pred_knowns_se <- rep.int(0,length(x))

  newasdf <- foreach (i=0:max(fold_list)) %dopar% par_gam(i,c,knowns[fold_list!=i,c],knowns[fold_list!=i,"time"],knowns,pred_knowns,pred_knowns_se)

  for (i in 0:max(fold_list)){
    print(newasdf)
    print(newasdf[[i+1]])
    pred_knowns[fold_list==i] <- newasdf[[i+1]][[1]]
    #pred_knowns_se[fold_list==i] <- newasdf[[i+1]][[2]]
  }

  pred_unknowns <- predict(b,data.frame(x=unknowns[,c]),type="response") #, se.fit = TRUE

  unknowns[paste(c,"+","RtGAM",sep="")] <- pred_unknowns #$fit
  knowns[paste(c,"+","RtGAM",sep="")] <- pred_knowns
}

write.csv(unknowns,"GAMpredTemp.csv",quote=T,row.names=F)
write.csv(knowns,"GAMtrainTemp.csv",quote=T,row.names=F)
