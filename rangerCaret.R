##################################################################
# already read data in from another file
##################################################################
# load packages
library(doParallel)
library(caret)
library(xgboost)

##################################################################
##################################################################

# set up the cross-validated hyper-parameter search
tgrid <- expand.grid(.mtry = 1:2,
                    .splitrule = "gini",
                    .min.node.size = c(5, 10, 20, 25, 30)
)

# set up function to perform cross-valdation on the models that are built
trcontrol = trainControl(method = "cv"
                       , number = 3
                       , verboseIter = TRUE
                       , returnData = FALSE
                       , summaryFunction=twoClassSummary	# Use AUC to pick the best model
                       , returnResamp = "all" # save losses across all models
                       , classProbs = TRUE # set to TRUE for AUC to be computedsummaryFunction = twoClassSummary
                       , allowParallel = TRUE
)

# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
rfTrain <- caret::train(x = as.matrix(x)
                        , y = y
                        , trControl = trcontrol
                        , tuneGrid = tgrid
                        , method = "ranger"
                        , metric = "ROC"
                        , num.trees = 500)
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

rfTrain$bestTune
plot(rfTrain)  		# Plot the performance of the training models
res <- rfTrain$results
res


