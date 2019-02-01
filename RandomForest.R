#this script is used to predict customer churn 
#the data was downloaded off of Kaggle

#load packages that I will need to
#read in data
#conduct exploratory analysis
#manipulate data
#build several models


#load necessary packages
library(dplyr)
library(reshape)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(RColorBrewer)
library(pROC)
library(data.table) 
library(randomForest)



VarSelectME<- function(checkDF){
  
  ### paramaters below are auto set - no need to manually code unless you would like to override
  N_trees <- 500 ## enter the number of Random Forest Trees you want built - if you are not sure 100 is ussually enough to converge nicely when doing variable selection only
  Recs<-length(Final[,1]) ## total records in the file
  Recs_Dep1 <-sum(Final$DEP)   ## how many total records where DEP=1 (dependent variable equals 1)
  Node_Size<-round(50/(Recs_Dep1/Recs)) ## auto calculation for min terminal node size so on average at least 50 DEP=1s are present in each terminal node '
  Max_Nodes<-NULL ## maximum number of terminal nodes.  20 to 25 is
  Sample_Size<-round(.3*Recs)  ## iteration sample size - 20% is usually good
  
  library("randomForest")
  set.seed(100)
  temp <- randomForest(checkDF[,indvarsc],checkDF$DEP
                       ,sampsize=c(Sample_Size),do.trace=TRUE,importance=TRUE,ntree=N_trees,replace=FALSE,forest=TRUE
                       ,nodesize=Node_Size,maxnodes=Max_Nodes,na.action=na.omit)
  
  RF_VARS <- as.data.frame(round(importance(temp), 2))
  RF_VARS <- RF_VARS[order(-RF_VARS$IncNodePurity) ,]  
  best_vi=as.data.frame(head(RF_VARS,N_Vars))
  
  topvars <-as.vector(row.names(best_vi))
  ## topvars now contains the top N variables
  
  return(topvars)
}
###################################################################################################


#####################################################################################
## function to split out the binary attributes
createBinaryDF <- function(depVar, checkDF){
  binaryCols <- c(depVar)
  nameVecBin<- names(checkDF)
  
  for (n in nameVecBin){
    if (n != depVar){
      checkBinary<-summary(checkDF[,n])
      # c("Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max.")
      isBinary<- ifelse((checkBinary["Min."] == 0 &   checkBinary["Max."] == 1
                         & (checkBinary["Median"]== 0 ||checkBinary["Median"]== 1)
                         & (checkBinary["1st Qu."]== 0 ||checkBinary["1st Qu."]== 1)
                         & (checkBinary["3rd Qu."]== 0 ||checkBinary["3rd Qu."]== 1)),"yes","no")
      
      
      
      if  (isBinary == "yes") {
        binaryCols<-  append(binaryCols, n)
        print(here<- paste("Adding binary: ",n ,sep=""))  
      }  
    } 
  }
  return(checkDF[,binaryCols])
}
#####################################################################################


######################################################################################
### Scaling function
scaleME <- function(checkDF){
  
  require(stats)
  
  ## center and scale the vars 
  checkDF<- as.data.frame(scale(checkDF),center=TRUE,scale=apply(seg,1,sd,na.rm=TRUE))
  
  ## take the cubed root
  cube_root<- function(x){x^1/3}
  checkDF<-as.data.frame(cube_root(checkDF))
  
  ## run softmax - convert all vars to a range of 0 to 1  
  ## 2 lines below do not work for some reason so needed to run the loop
  ## range01 <- function(x){(x-min(x))/(max(x)-min(x))}
  ## checkDF <- range01(checkDF) 
  nameVecBin<- names(checkDF)
  for (n in nameVecBin) {
    checkDF[,n]<-(checkDF[,n]-min(checkDF[,n]))/(max(checkDF[,n])-min(checkDF[,n]))
  }
  
  return(checkDF)
}
######################################################################################




library(reshape)
inputdf <- rename(trainchurnData, c(team_name="X"))
inputdf <- rename(trainchurnData, c(churn="DEP"))

## other options to import are the sas7bdat library (example below) and the R2SAS library
## useR2SAS once available - not available on the CRAN as of 1/20/2012
## library(sas7bdat)
## inputdf <- read.sas7bdat("L:/Work/SAS Data/ESPN/Jack/Ad Click Model/Ad_click_master_rl.sas7bdat")

## if you need to manually convert attributes below are some options
## inputdf$PurchDate <- as.Date(inputdf$PurchDate,"%m/%d/%Y")
## inputdf$WheelTypeID <- as.integer(inputdf$WheelTypeID)

names1 <- names(inputdf)
#fix(names1)



names2 <- c("genderFemale", "genderMale", "PartnerNo", "PartnerYes", "DependentsNo", 
            "DependentsYes", "PhoneServiceNo", "PhoneServiceYes", "MultipleLinesNo", 
            "MultipleLinesNophoneservice", "MultipleLinesYes", "InternetServiceDSL", 
            "InternetServiceFiberoptic", "InternetServiceNo", "OnlineSecurityNo", 
            "OnlineSecurityNointernetservice", "OnlineSecurityYes", "OnlineBackupNo", 
            "OnlineBackupNointernetservice", "OnlineBackupYes", "DeviceProtectionNo", 
            "DeviceProtectionNointernetservice", "DeviceProtectionYes", "TechSupportNo", 
            "TechSupportNointernetservice", "TechSupportYes", "StreamingTVNo", 
            "StreamingTVNointernetservice", "StreamingTVYes", "StreamingMoviesNo", 
            "StreamingMoviesNointernetservice", "StreamingMoviesYes", "ContractMonthtomonth", 
            "ContractOneyear", "ContractTwoyear", "PaperlessBillingNo", "PaperlessBillingYes", 
            "PaymentMethodBanktransferautomatic", "PaymentMethodCreditcardautomatic", 
            "PaymentMethodElectroniccheck", "PaymentMethodMailedcheck", "PhoneOnlyNo", 
            "PhoneOnlyYes", "InternetOnlyNo", "InternetOnlyYes", "PaperlessAutoPayNo", 
            "PaperlessAutoPayYes", "PhoneInternetNo", "PhoneInternetYes", 
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", 
            "DEP")



inputdf_1 <- inputdf[names2]
inputdf_1[is.na(inputdf_1)] <- 0


#summary(inputdf_1)
################################################################
## 2)split out the binary attributes

## see bottom section for the function called here (createBinaryDF)
droppedBinDF <-createBinaryDF("DEP", inputdf_1)
## droppedBindf contains the X and DEP variables as well

## now create the file with all non-binary attributes
delVar <- names(droppedBinDF)
## delVar <- delVar[delVar != "X"]    ## Keep X
## delVar <- delVar[delVar != "DEP"]  ## Keep DEP
mydropvars <- !((names(inputdf_1)) %in% (delVar))
inputdf2 <- inputdf_1[mydropvars]




################################################################
## 3)scale the non-binary attributes
## see bottom section for the function called here (scaleME)
#inputdf2 <-scaleME(inputdf2)



#inputdf3<- data.frame(scale(inputdf2, center=TRUE, scale=TRUE))



################################################################
## 4)Calculate transformations of non-binary attributes
## see bottom section for the function called here (scaleME)
#trans_vars <-transformME(inputdf2)





################################################################
## 5)Final File: combine all attributes
Final <- cbind(droppedBinDF, inputdf2)
summary(Final)

## drop all interim data
rm(droppedBinDF)
rm(trans_vars)
rm(inputdf2)
rm(delvar)
rm(mydropvars)








###################################################################################
## Build the Models: Random Forest                                               ##
###################################################################################


##################################################################################
## Create the independent variable string (indvarsc)
## save(Final, file = "Final.RData")
## load("Final.RData")

myvars <- names(Final) %in% c("Row.names","DEP")
tmp <- Final[!myvars]
indvarsc <- names(tmp)
rm(myvars)
rm(tmp)

rm(decile_report)
rm(auc_out)
rm(RF_VARS)
rm(pred)
rm(rank)


###################################################################################
## determine the top N variables to use using Random Forest - if you want to use all the variables skip this step
## you can think of this technique as similar tostepwise in regression

## manually enter the number of top variables selected you would like returned (must be <= total predictive vars)
N_Vars <- 15

### Run the variable selection procedure below.  The final top N variables will be returned
Top_N_Vars<-VarSelectME(Final)
names(Top_N_Vars)
fix(Top_N_Vars)

Top_N_Vars <- c("ContractMonthtomonth", "tenure", "OnlineSecurityNo", "TechSupportNo", 
                "InternetServiceFiberoptic", "TotalCharges", "MonthlyCharges", 
                "PaymentMethodElectroniccheck", "ContractTwoyear", "InternetServiceDSL", 
                "ContractOneyear", "OnlineBackupNo", "OnlineSecurityYes", 
                "PaperlessBillingYes")






###################################################################################
## build random forest model based on top N variables 

### paramaters below are auto set - no need to manually code unless you would like to override
N_trees <- 500 ## enter the number of Random Forest Trees you want built - if you are not sure 500 is usually enough to converge nicely when doing variable selection only
Recs<-length(Final[,1]) ## total records in the file
Recs_Dep1 <-sum(Final$DEP)   ## how many total records where DEP=1 (dependent variable equals 1)
Node_Size<-round(50/(Recs_Dep1/Recs)) ## auto calculation for min terminal node size so on average at least 50 DEP=1s are present in each terminal node '
Max_Nodes<-20 ## maximum number of terminal nodes.  20 to 25 is
Sample_Size<-round(.7*Recs)  ## iteration sample size - 20% is usually good

set.seed(100)
Final_RF<- randomForest(Final[,Top_N_Vars],Final$DEP
                        ,sampsize=c(Sample_Size)
                        ,do.trace=TRUE
                        ,importance=TRUE
                        ,ntree=N_trees
                        ,replace=FALSE
                        ,forest=TRUE
                        ,nodesize=Node_Size
                        ,maxnodes=Max_Nodes
                        ,na.action=na.omit)

RF_VARS <- as.data.frame(round(importance(Final_RF), 2))
RF_VARS <- RF_VARS[order(-RF_VARS$IncNodePurity) ,]

#save(Final_RF, file = "Final_RF_2011.RData")
#load("Final_RF_2011.RData")

# Score
pred <- data.frame(predict(Final_RF,Final[,Top_N_Vars]),type="prob") ##type= options are response, prob. or votes
pred <- pred[c(-2)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
library(gtools)
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.1), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(Final,pred,rank)

#Run AUC
library(caTools)
auc_out <- colAUC(Final_Scored$score, Final_Scored$DEP, plotROC=TRUE, alg=c("Wilcoxon","ROC"))

#Run Decile Report: do average of all model vars, avg DEP and min score, max score and avg score
library(sqldf)
decile_report <- sqldf("select rank, count(*) as qty, sum(DEP) as Responders, min(score) as min_score,
                       max(score) as max_score, avg(score) as avg_score
                       from Final_Scored
                       group by rank")

write.csv(decile_report,"decile_report.csv")

Final_Scored$score2 <- ifelse(Final_Scored$score > 0.78, 1, Final_Scored$score)


LogLoss <- function(DEP, score, eps=0.00001) {
  score <- pmin(pmax(score, eps), 1-eps)
  -1/length(DEP)*(sum(DEP*log(score)+(1-DEP)*log(1-score)))
}
LogLoss(Final_Scored$DEP,Final_Scored$score)
