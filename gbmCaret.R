##################################################################
# already read data in from another file
##################################################################
# load packages
library(doParallel)
library(caret)

##################################################################
##################################################################

y <- as.factor(make.names(Final$churn))
x <- Final[,-54]

gbm_trcontrol = caret::trainControl(method = "cv"
                                    , number = 5
                                    , verboseIter = TRUE
                                    , returnData = FALSE
                                    , summaryFunction=twoClassSummary	# Use AUC to pick the best model
                                    , returnResamp = "all" # save losses across all models
                                    , classProbs = TRUE
                                    )

gbmGrid <-  expand.grid(interaction.depth = c(2), 
                        n.trees = c(100), 
                        shrinkage = c(0.1),
                        n.minobsinnode = c(30)
                        )

nrow(gbmGrid)

set.seed(420)
# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()

gbmFit <- caret::train(x = x
                       , y = y
                       , method = "gbm"
                       , metric = "ROC"
                       , verbose = FALSE
                       ## Now specify the exact models 
                       ## to evaluate:
                       , tuneGrid = gbmGrid
                       , trControl = gbm_trcontrol
                       )

t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

gbmFit$bestTune
plot(gbmFit)  		# Plot the performance of the training models
res <- gbmFit$results
res

gbmFit