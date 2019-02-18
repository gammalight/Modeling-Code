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
library(caTools)
library(gtools)
library(sqldf)
library(gbm)


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

#Calculate Logloss
LogLoss <- function(DEP, score, eps=0.00001) {
  score <- pmin(pmax(score, eps), 1-eps)
  -1/length(DEP)*(sum(DEP*log(score)+(1-DEP)*log(1-score)))
}
######################################################################################


#read in the customer churn csv data file
churnData <- read.csv("C:\\Users\\Kevin Pedde\\Documents\\R\\Work\\CustomerChurn\\customerchurn\\TelcoCustomerChurn.csv")

###############################################################
### Feature Engineering ###
#will get to this later, just want to build some models to get baseline
#ideas for new variables:
# - Phone only
# - Internet only
# - paperless billing and auto pay

#Okay lets create these variables
churnData <- churnData %>%
  mutate(PhoneOnly = if_else(PhoneService == 'Yes' & InternetService == 'No', 'Yes', 'No'),
         InternetOnly = if_else(PhoneService == 'No' & InternetService != 'No', 'Yes', 'No'),
         PhoneInternet = if_else(PhoneService == 'Yes' & InternetService != 'No', 'Yes', 'No'),
         PaperlessAutoPay = if_else(PaperlessBilling == 'Yes' & 
                                      PaymentMethod %in% c("Bank transfer (automatic)","Credit card (automatic)"), 'Yes', 'No'),
         churn = if_else(Churn == 'Yes',1,0))


#first drop all tenure 0 people
churnData <- churnData %>%
  select(-customerID) %>% #deselect CustomerID
  filter(tenure > 0) %>%
  droplevels()

## Create Dummy Variables ##
dmy <- dummyVars(" ~ gender + Partner + Dependents + PhoneService +
                 MultipleLines + InternetService + OnlineSecurity +
                 OnlineBackup + DeviceProtection + TechSupport +
                 StreamingTV + StreamingMovies + Contract + PaperlessBilling +
                 PaymentMethod + PhoneOnly + InternetOnly + PaperlessAutoPay +
                 PhoneInternet", 
                 data = churnData,
                 fullRank = FALSE)
dmyData <- data.frame(predict(dmy, newdata = churnData))
#print(head(dmyData))
#strip the "." out of the column names
colClean <- function(x){ colnames(x) <- gsub("\\.", "", colnames(x)); x } 
dmyData <- colClean(dmyData) 

#lets combine the new dummy variables back with the original continuous variables
churnDataFinal <- cbind(dmyData, churnData[,c(2,5,18,19,25)])

#lets get a traing and test data set using the createPartition function from Caret
set.seed(420)
inTrain <- createDataPartition(churnDataFinal$churn, p = 9/10, list = FALSE, times = 1)
trainchurnData <- churnDataFinal[inTrain,]
testchurnData <- churnDataFinal[-inTrain,]


inputdf <- rename(trainchurnData, c(churn="DEP"))
inputdf_test <- rename(testchurnData, c(churn="DEP"))

#Get names of columns
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

inputdf_test1 <- inputdf_test[names2]
inputdf_test1[is.na(inputdf_test1)] <- 0


#summary(inputdf_1)
################################################################
## 2)split out the binary attributes

## see bottom section for the function called here (createBinaryDF)
## droppedBindf contains the X and DEP variables as well
droppedBinDF <-createBinaryDF("DEP", inputdf_1)
droppedBinDF_test <-createBinaryDF("DEP", inputdf_test1)

## now create the file with all non-binary attributes
delVar <- names(droppedBinDF)
## delVar <- delVar[delVar != "X"]    ## Keep X
## delVar <- delVar[delVar != "DEP"]  ## Keep DEP
mydropvars <- !((names(inputdf_1)) %in% (delVar))
inputdf2 <- inputdf_1[mydropvars]
inputdf_test2 <- inputdf_test1[mydropvars]




################################################################
## 5)Final File: combine all attributes
Final <- cbind(droppedBinDF, inputdf2)
Final_test <- cbind(droppedBinDF_test, inputdf_test2)

#Final$DEP <- as.factor(as.character(Final$DEP))
#Final_test$DEP <- as.factor(as.character(Final_test$DEP))

summary(Final_test)
summary(Final)

## drop all interim data
rm(droppedBinDF)
rm(trans_vars)
rm(inputdf2)
rm(delvar)
rm(mydropvars)




###################################################################################
## Build the Models: GBM                                                         ##
###################################################################################



gbmModel <- gbm(DEP ~ ., 
                data = Final, 
                #var.monotone = c(0, 0, 0, 0, 0, 0),
                distribution = "bernoulli", 
                n.trees = 100, 
                shrinkage = 0.1,
                interaction.depth = 3, 
                bag.fraction = 0.5, 
                train.fraction = 0.5,
                n.minobsinnode = 10, 
                cv.folds = 5, 
                keep.data = TRUE,
                verbose = TRUE, 
                n.cores = 4)



# Check performance using the out-of-bag (OOB) error; the OOB error typically
# underestimates the optimal number of iterations
best.iter <- gbm.perf(gbmModel, method = "OOB")
print(best.iter)
# Check performance using the 50% heldout test set
best.iter <- gbm.perf(gbmModel, method = "test")
print(best.iter)
# Check performance using 5-fold cross-validation
best.iter <- gbm.perf(gbmModel, method = "cv")
print(best.iter)
# Plot relative influence of each variable
par(mfrow = c(1, 2))
summary(gbmModel, n.trees = 1) # using first tree
summary(gbmModel, n.trees = best.iter) # using estimated best number of trees
# Compactly print the first and last trees for curiosity
print(pretty.gbm.tree(gbmModel, i.tree = 1))
print(pretty.gbm.tree(gbmModel, i.tree = gbmModel$n.trees))



#Now take the significant variables and build models and test using holdout sample
namesFinal <- c("ContractMonthtomonth","tenure","InternetServiceFiberoptic",
                "TotalCharges","TechSupportNo","MonthlyCharges","OnlineSecurityNo",
                "PaymentMethodElectroniccheck","ContractTwoyear","PaperlessBillingYes",
                "OnlineBackupNo","MultipleLinesNo",
                "PaperlessBillingNo","DeviceProtectionNo","DEP")

TopnamesFinal <- c("ContractMonthtomonth","tenure","InternetServiceFiberoptic",
                   "TotalCharges","TechSupportNo","MonthlyCharges","OnlineSecurityNo",
                   "PaymentMethodElectroniccheck","ContractTwoyear","PaperlessBillingYes",
                   "OnlineBackupNo","MultipleLinesNo",
                   "PaperlessBillingNo","DeviceProtectionNo")
#Do a little more variable selection based on variable importance from the below model
#"MultipleLinesYes",
#"SeniorCitizen",
#"StreamingTVYes",
#"PartnerYes",
#"PhoneServiceYes",
#"genderMale",
#"OnlineSecurityYes",
#"StreamingMoviesYes",
#"genderFemale",
#"PaperlessAutoPayNo",
#"PaymentMethodMailedcheck",
FinalModelData <- Final[namesFinal]
FinalModelData$DEP <- as.factor(as.character(FinalModelData$DEP))

gbmModelFinal <- gbm(DEP ~ ., 
                data = FinalModelData, 
                #var.monotone = c(0, 0, 0, 0, 0, 0),
                distribution = "bernoulli", 
                n.trees = 40, 
                shrinkage = 0.1,
                interaction.depth = 3, 
                bag.fraction = 0.5, 
                train.fraction = 0.5,
                n.minobsinnode = 10, 
                cv.folds = 5, 
                keep.data = FALSE,
                verbose = TRUE, 
                n.cores = 4)

summary.gbm(gbmModelFinal,
            cBars = length(gbmModelFinal$var.names),
            n.trees = gbmModelFinal$n.trees, 
            plotit = TRUE, 
            order = TRUE,
            method = relative.influence, 
            normalize = TRUE)




# Score
pred <- data.frame(predict.gbm(gbmModelFinal,newdata = FinalModelData[TopnamesFinal]),
                   type="prob") ##type= options are response, prob. or votes
pred <- pred[c(-2)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.1), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(Final,pred,rank)

#Run AUC
#use the two different ways
auc_out <- colAUC(Final_Scored$score, Final_Scored$DEP, plotROC=TRUE, alg=c("Wilcoxon","ROC"))
rocObj <- roc(Final_Scored$DEP, Final_Scored$score)
auc(rocObj)

#Run Decile Report: do average of all model vars, avg DEP and min score, max score and avg score
decile_report <- sqldf("select rank, count(*) as qty, sum(DEP) as Responders, min(score) as min_score,
                       max(score) as max_score, avg(score) as avg_score
                       from Final_Scored
                       group by rank")

write.csv(decile_report,"decile_report.csv")

#Calculate the Logloss metric
LogLoss(Final_Scored$DEP,Final_Scored$score)
#0.4132963
