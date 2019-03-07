##################################################################
# already read data in from another file
##################################################################
# load packages
library(doParallel)
library(caret)
library(xgboost)

##################################################################
##################################################################


y <- as.factor(make.names(Final$DEP))
x <- Final[,-1]

# Model to predict workers 
# pack the training control parameters
xgb_trcontrol = caret::trainControl(method = "cv"
                                   , number = 7
                                   , verboseIter = TRUE
                                   , returnData = FALSE
                                   , summaryFunction=twoClassSummary	# Use AUC to pick the best model
                                   , returnResamp = "all" # save losses across all models
                                   , classProbs = TRUE
                                   )

#############################################################################
# not how you want to tune a model
# but since the data is small i figured id let it run
xgb_trtune <- expand.grid(nrounds = 500
                        , subsample = c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
                        , eta = c(0.01, 0.1)
                        , max_depth = c(20, 25, 30)
                        , gamma = c(1, 2) 
                        , colsample_bytree = c(0.4, 0.6, 0.8) 
                        , min_child_weight = c(1, 2, 3, 4, 5, 6)
                        )


# Set up to do parallel processing
no_cores <- detectCores()  # Number of cores
cl <- makeCluster(no_cores - 2)
registerDoParallel(no_cores)		# Registrer a parallel backend for train
getDoParWorkers()

t1 <- Sys.time()
xgbTrain <- train(x = as.matrix(x)
                  , y = y
                  , trControl = xgb_trcontrol
                  , tuneGrid = xgb_trtune
                  , method = "xgbTree"
                  , metric = "ROC")
t2 <- Sys.time()
print(t2 - t1)
stopCluster(cl)

xgbTrain$bestTune
plot(xgbTrain)  		# Plot the performance of the training models
res <- xgbTrain$results
res


# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")
