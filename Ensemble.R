# ENSEMBLE

# Bagging

library(randomForest)

model_bag <- randomForest(formula = cnt ~ .,
                               data=data_training, 
                               mtry=(ncol(data_training)-1), ntree=500)
print(model_bag)

# Random Forest

#  Restricts the number of random variables (columns) considered on every split to 22/3=7

model_rf <- randomForest(cnt ~ ., data=data_training, ntree=500)
print(model_rf)

# set the number of random inputs selected at each split
tunerf <- tuneRF(x=data_training[,-"cnt"], y=data_training[,cnt], ntreeTry = 500)

# Gradient Boosting

library(gbm)

model_gbm <- gbm(cnt ~ ., 
                 data=data_training, 
                 distribution="gaussian",
                 interaction.depth=1,
                 n.trees=500, 
                 shrinkage=0.1,
                 cv.folds = 5)
print(model_gbm)

# Estimates the optimal number of boosting iterations and plot the performance measures
gbm.perf(model_gbm, method = "cv")
# Training error in black, validation error in green, the blue dashed line shows what is the optimal number of iterations according to the metric and validation procedure used


#------------------------------------------------------

# Model Evaluation

# Test Error

# Predict on the test set, then plot the test set RMSE against an increasing ensemble size

# Size of ensembles
ntrees <- 500

# Lists to collect results
model_bag_rmse <- vector("list", ntrees)
model_rf_rmse <- vector("list", ntrees)
model_gbt_rmse <- vector("list", ntrees)

# One-time models
model_tree_training <- rpart(cnt ~ ., data=data_training, method="anova")
model_tree_rmse <- sqrt(mean((data_test$cnt - 
                                predict(model_tree_training, newdata = data_test))^2)) # 76.66

sqrt(mse_tree) # rmse obtain from the whole data 73.4

model_gbt <- gbm(cnt ~ ., data=data_training, distribution='gaussian', 
                 n.trees=ntrees, shrinkage=0.05, interaction.depth=1)

# Loop for MSE  
for(i in 1:ntrees){
  
  model_bag_training <- randomForest(cnt ~ ., data=data_training, 
                            mtry=(ncol(data_training)-1), ntree=i) 
  model_bag_rmse[[i]] <- sqrt(mean((data_test$cnt - 
                                      predict(model_bag_training, newdata = data_test))^2))
  
  model_rf_training <- randomForest(cnt ~ ., data=data_training, ntree=i) 
  model_rf_rmse[[i]] <- sqrt(mean((data_test$cnt - 
                                     predict(model_rf_training, newdata = data_test))^2)) 
  
  model_gbt_rmse[[i]] <- sqrt(mean((data_test$cnt - 
                                      predict(model_gbt, newdata = data_test, n.trees=i))^2))
}

# Organize results
dt_results <- data.table(
  "n" = 1:ntrees, 
  "BoostedTrees" = unlist(model_gbt_rmse),
  "RandomForest"= unlist(model_rf_rmse), 
  "Bagging" = unlist(model_bag_rmse), 
  "SingleTree" = rep(model_tree_rmse,ntrees))

# Plot results
ggplot(data = dt_results, aes(x=n)) +
  geom_line(aes(y=SingleTree), color="#000000", linetype="dashed") +
  geom_line(aes(y=Bagging), color="#d3d3d3") +
  geom_line(aes(y=RandomForest), color="#767676") +
  geom_line(aes(y=BoostedTrees), color="#000000") +
  labs(
    x = "Number of trees",
    y = "RMSE",
    title="Ensemble Performances") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"),
        axis.ticks=element_blank())

# OOB Error

ntrees <- 500

model_rf_rmse_test <- vector("list", ntrees)
model_rf_rmse_train <- vector("list", ntrees)
model_rf_rmse_oob <- vector("list", ntrees)

for(i in 1:ntrees){
  model_rf_training <- randomForest(cnt ~ ., data=data_training, ntree=i)
  #OOB error
  model_rf_rmse_oob[[i]] <- sqrt(tail(model_rf_training$mse, 1))
  #test error
  model_rf_rmse_test[[i]] <- sqrt(mean((data_test$cnt - 
                                          predict(model_rf_training, newdata = data_test))^2)) 
  #training error
  model_rf_rmse_train[[i]] <- sqrt(mean((data_training$cnt - 
                                           predict(model_rf_training, newdata = data_training))^2)) 
}
# Create table of results by converting a list to a vector
dt_results <- data.table("n" = 1:ntrees, 
                         "Test" = unlist(model_rf_rmse_test),
                         "Train"= unlist(model_rf_rmse_train), 
                         "OOB" = unlist(model_rf_rmse_oob))

ggplot(data = dt_results, aes(x=n)) +
  geom_line(aes(y=Test), color="#4CA7DE") +
  geom_line(aes(y=Train), color="#f1b147") +
  geom_line(aes(y=OOB), color="#000000", linetype="dashed") +
  labs(
    x = "Number of trees",
    y = "RMSE") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"),
        axis.ticks=element_blank())

#------------------------------------------------------

# Hyperparameter Search

  # Random Forest
library(caret)
# Partition our data into folds so every parameter combination is validated against the same fold assignments
folds <- createFolds(data_training[,cnt], k = 5)

# Set up the grid for our loop, which now includes combinations of different ensemble sizes and minimum leaf sizes
loop_grid <- expand.grid(ntrees = c(10,50,100),
                         nodesize = c(1,5,10))

# Set up the number of random inputs on each split mtry
rf_grid <-  expand.grid(mtry = 1:(ncol(data_training)-1))
train_settings <- trainControl(method = "cv", 
                               number = 5,
                               index = folds)
results <- list()

for(i in 1:nrow(loop_grid)){
  ntrees <- loop_grid[i,1]
  nodesize <- loop_grid[i,2]
  caret_rf <- train(cnt ~ ., 
                    data = data_training,
                    method = "rf",
                    ntrees = ntrees,
                    nodesize = nodesize,
                    trControl = train_settings,
                    tuneGrid = rf_grid)
  
  results[[i]] <- caret_rf
}

names(results) <- paste("ntrees_", loop_grid$ntrees,
                        "nodesize_", loop_grid$nodesize)

summary(resamples(results))$statistics$RMSE

  # Boosting
train_settings <- trainControl(method = "cv", number = 5)

gbm_grid =  expand.grid(interaction.depth = 1:5,
                        n.trees = c(500,1000,2000,5000),
                        shrinkage = c(0.1, 0.01, 0.001),
                        n.minobsinnode = c(10,20))

caret_gbm <- train(RetailPrice ~ ., data = dt_training,
                   method = "gbm",
                   trControl = train_settings,
                   tuneGrid = gbm_grid,
                   verbose = FALSE)