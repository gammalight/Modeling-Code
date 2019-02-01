
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

#################################
## try using the recursive feature elimination function in caret ##
## this performs backward selection method to pick the best n variables ##
## this is based on predictor importance ranking
#################################
subsets <- c(1:25)
ctrl <- rfeControl(functions = lmFuncs,
                   method = "repeatedcv",
                   repeats = 10,
                   verbose = FALSE)

y <- as.factor(as.character(churnDataFinal$churn))
x <- churnDataFinal[,-54]

lmProfile <- rfe(x,
                 y,
                 metric = "Accuracy",
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile
names(lmProfile$fit$model)

trellis.par.set(caretTheme())
plot(lmProfile, type = c("g", "o"))

#top 5 variables
#InternetServiceFiberoptic
#InternetServiceDSL
#PaperlessBillingNo
#ContractMonthtomonth
#PhoneServiceNo
#StreamingMoviesNo (with accuracy as the metric)
#StreamingTVNo
#PhoneServiceNo
#PaperlessAutoPayNo
#PaymentMethodElectroniccheck
#MultipleLinesNo
#TechSupportNo
#SeniorCitizen
#OnlineSecurityNo
#PaymentMethodBanktransferautomatic
#ContractOneyear
#PaymentMethodCreditcardautomatic
#DependentsNo
#OnlineBackupNo
#DeviceProtectionNo
#genderFemale
#PartnerNo
#tenure
#MonthlyCharges
#TotalCharges


## lets use bag ##

subsets <- c(1:20)
ctrl <- rfeControl(functions =  treebagFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

y <- as.factor(as.character(churnDataFinal$churn))
x <- churnDataFinal[,-54]

treebagProfile <- rfe(x,
                      y,
                      metric = "Accuracy",
                      sizes = subsets,
                      rfeControl = ctrl)

treebagProfile
names(treebagProfile$fit$X)

trellis.par.set(caretTheme())
plot(lmProfile, type = c("g", "o"))

#TotalCharges
#MonthlyCharges
#tenure
#OnlineSecurityNo
#TechSupportNo
#ContractMonthtomonth
#InternetServiceFiberoptic
#PaymentMethodElectroniccheck
#genderFemale
#genderMale
#PartnerNo
#PaperlessBillingNo
#OnlineBackupNo
#PartnerYes
#DependentsNo
#PaperlessBillingYes
#MultipleLinesNo
#SeniorCitizen
#DependentsYes
#DeviceProtectionNo
#
