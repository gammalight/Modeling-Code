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
library(xgboost)


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
inTrain <- createDataPartition(churnDataFinal$churn, p = 0.5, list = FALSE, times = 1)
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
            "SeniorCitizen",# "tenure", "MonthlyCharges", "TotalCharges", 
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

X <- Final
XX <- Final_test

#y <- as.factor(as.character(Final$DEP))
y <- as.numeric(as.character(Final$DEP))
yy <- as.numeric(as.character(Final_test$DEP))


Top_N_Vars <- c("ContractMonthtomonth", "genderFemale", "PartnerNo", "OnlineBackupNo", 
                "PaperlessBillingNo", "OnlineSecurityNo", "PaymentMethodElectroniccheck", 
                "StreamingTVNo", "DeviceProtectionNo", "MultipleLinesNo", "SeniorCitizen")#, 
                "TechSupportNo", "DependentsNo", "StreamingMoviesNo", "InternetServiceFiberoptic", 
                "InternetServiceDSL", "MultipleLinesYes", "PaymentMethodBanktransferautomatic", 
                "StreamingMoviesYes", "genderMale", "PartnerYes", "PaymentMethodMailedcheck", 
                "DeviceProtectionYes", "PaymentMethodCreditcardautomatic", "StreamingTVYes"
)



xgb <- xgboost(data = data.matrix(X[,Top_N_Vars]), 
               label = y, 
               eta = 0.1,
               max_depth = 20, 
               nrounds=500, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 6,
               eval_metric = "auc",
               objective = "binary:logistic",
               #num_class = 12,
               nthread = 6
)


model <- xgb.dump(xgb, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model

# Get the feature real names
names <- dimnames(data.matrix(X[,-1]))[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:20,])

Top_N_Vars <- importance_matrix$Feature[1:25]
fix(Top_N_Vars)


# Score
pred <- data.frame(predict(xgb,data.matrix(X[,Top_N_Vars]),type="prob")) ##type= options are response, prob. or votes
pred <- pred[c(-2)]
names(pred) <- "score"
summary(pred)

# Apply Deciles
# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-pred$score, q=seq(0, 1, 0.125), labels=F))
names(rank) <- "rank"

# apply the rank
Final_Scored <- cbind(X,pred,rank)

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

#find the Youden index to use as a cutoff for cunfusion matrix
coords(rocObj, "b", ret="t", best.method="youden") # default
#0.3083464

#Classify row as 1/0 depending on what the calculated score is
#play around with adjusting the score to maximize accuracy or any metric
Final_Scored <- Final_Scored %>%
  mutate(predClass = if_else(score > 0.51, 1, 0),
         predClass = as.factor(as.character(predClass)),
         DEPFac = as.factor(as.character(DEP)))

#Calculate Confusion Matrix
confusionMatrix(data = Final_Scored$predClass, 
                reference = Final_Scored$DEPFac)



###################################
###################################

#Now score the holdout sample and find all the metrics (AUC, Logloss, etc)

#now predict on the test set and compare results
preds_test <- data.frame(predict(xgb,data.matrix(XX[,Top_N_Vars]),type="prob"))
preds_test <- preds_test[c(-2)]
names(preds_test) <- "target"
summary(preds_test)

# 0.1 option makes 10 equal groups (.25 would be 4).  negative option (-pred$score) makes the highest score equal to 1
rank <- data.frame(quantcut(-preds_test$target, q=seq(0, 1, 0.125), labels=F))
names(rank) <- "rank"

FinalScored_test <- cbind(XX, preds_test) #, rank)
auc_out <- colAUC(FinalScored_test$target, FinalScored_test$DEP, plotROC=TRUE, alg=c("Wilcoxon","ROC"))
rocObj <- roc(FinalScored_test$DEP, FinalScored_test$target)
auc(rocObj)
0.6977

write.csv(FinalScored_test
          ,"submission.csv"
          ,row.names=FALSE)


#FinalScored_test <- FinalScored_test %>%
#  mutate(DEPTemp = ifelse(DEP == "X0", 0, 1))
#Calculate the Logloss metric
LogLoss(FinalScored_test$DEP,FinalScored_test$score)
#0.4607649

#find the Youden index to use as a cutoff for cunfusion matrix
coords(rocObj, "b", ret="t", best.method="youden") # default
#0.3126991

#Classify row as 1/0 depending on what the calculated score is
#play around with adjusting the score to maximize accuracy or any metric
FinalScored_test <- FinalScored_test %>%
  mutate(predClass = if_else(score > 0.51, 1, 0),
         predClass = as.factor(as.character(predClass)),
         DEPFac = as.factor(as.character(DEP)))

#Calculate Confusion Matrix
confusionMatrix(data = FinalScored_test$predClass, 
                reference = FinalScored_test$DEPFac)






##################################################################
##################################################################
# Set up to do parallel processing   
registerDoParallel(6)		# Registrer a parallel backend for train
getDoParWorkers()

xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol_1
                  , tuneGrid = expand.grid(nrounds = 500
                                           , subsample = 0.5
                                           , eta = c(0.01, 0.1)
                                           , max_depth = c(5,10,15)
                                           , gamma = c(1, 2) 
                                           , colsample_bytree = c(0.4, 0.7, 1.0) 
                                           , min_child_weight = c(0.5, 1, 1.5))
                  , method = "xgbTree"
                  , metric = "ROC")

gbm.tune$bestTune
plot(gbm.tune)  		# Plot the performance of the training models
res <- gbm.tune$results
res


# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")


