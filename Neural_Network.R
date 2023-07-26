# Neural Network

source("Data preparation.R")
source("Linear_model.R")

library(data.table)
library(ggplot2)
library(keras)
library(plot3D)
library(viridisLite)
library(tensorflow)
library(reticulate)
library(tidyverse)
library(tidymodels)
library(caret)
library(neuralnet)


# PREPARING THE DATA

# Define dependent and independent variables
  train_features <- data_training %>% select(-cnt) 
  train_labels <- data_training %>% select(cnt)  # dependent (label)
  
  test_features <- data_test %>% select(-cnt)
  test_labels <- data_test %>% select(cnt)

# Normalize to avoid scale issues first on train and then test dataset
  normalizer <- layer_normalization(axis = -1)  # -1 so all features are normalized independently
  
  normalizer %>% adapt(as.matrix(train_features)) # apply layer on feature space
  
  test_normalizer <- layer_normalization(axis = -1)
  
  test_normalizer %>% adapt(as.matrix(test_features))
  
  ####################### INTERMEDIATE STEPS FOR UNDERSTANDING##################
  
  # ONE FEAUTURE LINEAR REGRESSION WITH KERAS
  
    # First create normalized feature matrix
    temp <- matrix(train_features$temp)  # create matrix
    temp_normalizer <- layer_normalization(input_shape = shape(1), axis = NULL) # set normalization layer
    temp_normalizer %>% adapt(temp)  # adapt feature matrix to layer
    
    # setting up temp one layer model
    temp_model <- keras_model_sequential() %>%
      temp_normalizer() %>%
      layer_dense(units = 1)
    
    summary(temp_model)
    
    # Predict the first 30 temp values with the untrained model
    predict(temp_model, temp[1:30,])
    
    # so far no optimization at place - set model with mean_squared_error
    temp_model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.1),
      loss = 'mean_squared_error'
    )
    
    # Keras fit() function will now estimate 50 epochs
    history <- temp_model %>% fit(
      as.matrix(train_features$temp),
      as.matrix(train_labels),
      epochs = 50,
      # Suppress logging.
      verbose = 0,
      # Calculate validation results on 20% of the training data.
      validation_split = 0.2
    )
    
    plot(history)
    
    # Now as only two-dimensional model we can plot the results
    x <- seq(0, 1, length.out = 1.01)
    y <- predict(temp_model, x)
    
    ggplot(data) +
      geom_point(aes(x = temp, y = cnt, color = "data")) +
      geom_line(data = data.frame(x, y), aes(x = x, y = y, color = "prediction"))
  
    test_results <- list()
    test_results[["temp_model"]] <- temp_model %>% evaluate(
      as.matrix(test_features$temp),
      as.matrix(test_labels),
      verbose = 0
    )
  
  # MULTI FEATURE LINEAR REGRESSION
    linear_model <- keras_model_sequential() %>%
      normalizer() %>%
      layer_dense(units = 1)
    
    predict(linear_model, as.matrix(train_features[1:10, ]))
    
    linear_model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.1),
      loss = 'mean_squared_error'
    )
    
    # Similar to above only with whole feature-space
    history <- linear_model %>% fit(
      as.matrix(train_features),
      as.matrix(train_labels),
      epochs = 50,
      # Suppress logging.
      verbose = 0,
      # Calculate validation results on 20% of the training data.
      validation_split = 0.2
    )
    
    # Plot loss history
    plot(history)
    
    # Collect results on test data
    test_results[['linear_model']] <- linear_model %>%
      evaluate(
        as.matrix(test_features),
        as.matrix(test_labels),
        verbose = 0
      )


############################## DEEP NEURAL NETWORKS#############################
  # 1. Single feature input model
  # 2. Multi feature input model (MAIN MODEL)
  # 3. Varying learning rate over single feature input model
  # 4. Re-run model on PCA Data

# Setting up the model
  build_and_compile_model <- function(norm) {
    model <- keras_model_sequential() %>%
      norm() %>%
      layer_dense(16, activation = 'relu') %>% # DNN with 3 layers 
      layer_dense(16, activation = 'relu') %>% # and width of 16 neurons
      layer_dense(16, activation = 'relu') %>%
      layer_dense(1)
    
    model %>% compile(
      loss = 'mean_absolute_error',           # Attention: absolute loss here (alt. squared)
      optimizer = optimizer_adam(0.001)         # Learning rate of the GD
    )
    
    model
  }

# 1. SINGLE INPUT DEEP NEURAL NETWORK
  dnn_temp_model <- build_and_compile_model(temp_normalizer)
  
  summary(dnn_temp_model)
  
  history <- dnn_temp_model %>% fit(
    as.matrix(train_features$temp),
    as.matrix(train_labels),
    validation_split = 0.2,
    verbose = 0,
    epochs = 100    # 50 epochs doesnt require that much computation and sufficient to bottom out loss
  )
  
  plot(history)
  
  # As its only 2-D it's easy to plot the prediction
  x <- seq(0.0, 1, length.out = 1.01)
  y <- predict(dnn_temp_model, x)
  
  ggplot(data) +
    geom_point(aes(x = temp, y = cnt, color = "data")) +
    geom_line(data = data.frame(x, y), aes(x = x, y = y, color = "prediction"))
  
  # Collect results on test data
  test_results[['dnn_temp_model']] <- dnn_temp_model %>% evaluate(
    as.matrix(test_features$temp),
    as.matrix(test_labels),
    verbose = 0
  )
    
        ########################### MAIN MODEL ##############################
  
# 2. MULTI INPUT DEEP NEURAL NETWORK
  dnn_model <- build_and_compile_model(normalizer)   # store new model with same model-set up 
                                                     # but other Input Matrix
  dnn_history <- dnn_model %>% fit(        # Run model with fit command
    as.matrix(train_features),
    as.matrix(train_labels),
    validation_split = 0.2,
    verbose = 0,
    epochs = 100
  )
  
  plot(dnn_history)
  
  test_results[['dnn_model']] <- dnn_model %>% evaluate(
    as.matrix(test_features),
    as.matrix(test_labels),
    verbose = 0
  )

            ###########################################################  
  
# 3. DEEP NEURAL NETWORK WITH VARYING LEARNING RATES

# APPROACH:
  # To inspect on varying settings of the learning rate on model performance we set-up 
  # a loop within the deep neural netweork is trained on the single "temp" input (to 
  # reduce computation only "temp") 
  # Therefore: 
  # 1. specify model with option of varying optimizer settings
  # 2. Loop different DNNs on the training data and saver results in "training_histories"
  # 3. Plot results in "training_histories" with a little plot loop 

# 3.1 Function to build and compile the model with a given normalization and learning rate
  build_and_compile_model <- function(norm, learning_rate) {
    model <- keras_model_sequential() %>%
      norm() %>%
      layer_dense(16, activation = 'relu') %>%
      layer_dense(16, activation = 'relu') %>%
      layer_dense(16, activation = 'relu') %>%
      layer_dense(1)
    
    model %>% compile(
      loss = 'mean_absolute_error',
      optimizer = optimizer_adam(learning_rate) # Set the learning rate here
    )
    
    model
  }

# List of learning rates 
  learning_rate <- c(0.1, 0.01, 0.001)

# List to store training histories
  training_histories <- list()
  
# 3.2 Loop through each learning rate and train the model
  for (lr in learning_rate) {
    # Create the model with the specified learning rate
    dnn_var_model <- build_and_compile_model(temp_normalizer, lr)
    
    # Train the model with training data
    history <- dnn_var_model %>% fit(
      as.matrix(train_features$temp),    # only temp as example input to reduce computation
      as.matrix(train_labels),
      validation_split = 0.2,
      verbose = 0,
      epochs = 50
    )
    
    # Store the training history in the list
    training_histories[[as.character(lr)]] <- history
  }


# 3.3 Plot all the training and validation losses together

# Save the plot as a PDF file
pdf("training_rates_plots.pdf")

  par(mfrow = c(1, length(learning_rate)))  # Plotting layout - arranges 3 plots side by side
  for (i in seq_along(learning_rate)) {
    lr <- learning_rate[i]
    history <- training_histories[[as.character(lr)]]
    
    plot(history$metrics$loss, type = "l", col = "blue", main = paste("Learning Rate =", lr))
    lines(history$metrics$val_loss, col = "red")
    legend("topright", c("Training Loss", "Validation Loss"), col = c("blue", "red"), lty = 1)
  }
  par(mfrow = c(1, 1))  # Reset the plotting layout to default

# close pdf file
dev.off()


# 4. RERUN MODEL on PCA dimensionality reduced data
  pca_training_sample <- data_PC18[,sample(.N, floor(.N*.80))]
  pca_data_training <- data_PC18[pca_training_sample]
  pca_data_test <- data_PC18[-pca_training_sample]
  
  pca_train_features <- pca_data_training %>% select(-cnt)
  pca_train_labels <- pca_data_training %>% select(cnt)
  
  pca_test_features <- pca_data_test %>% select(-cnt)
  pca_test_labels <- pca_data_test %>% select(cnt)

# Same procedure with other data and no normalization layer as pca data already scaled
  build_and_compile_model <- function(norm) {
    model <- keras_model_sequential() %>%
      layer_dense(16, activation = 'relu') %>% # DNN with 3 layers 
      layer_dense(16, activation = 'relu') %>% # and width of 16 neurons
      layer_dense(16, activation = 'relu') %>%
      layer_dense(1)
    
    model %>% compile(
      loss = 'mean_absolute_error',           # Attention: absolute loss here (alt. squared)
      optimizer = optimizer_adam(0.001)         # Learning rate of the GD
    )
    
    model
  }
  
  dnn_pca_model <- build_and_compile_model()   # store new model with same model-set up 
  # but other Input Matrix
  pca_history <- dnn_pca_model %>% fit(        # Run model with fit command
    as.matrix(pca_train_features),
    as.matrix(pca_train_labels),
    validation_split = 0.2,
    verbose = 0,
    epochs = 100
  )

# Compare loss reduction
  par(mfrow = c(1, 2))  # Plotting layout - arranges 3 plots side by side
  
  plot(pca_history)
  plot(dnn_history)
  par(mfrow = c(1, 1))  # Plotting layout - arranges 3 plots side by side
  
  test_results[['dnn_pca_model']] <- dnn_pca_model %>% evaluate(
    as.matrix(pca_test_features),
    as.matrix(pca_test_labels),
    verbose = 0
  )


########################### VALIDATION AND PERFORMANCE ########################  


# PERFORMANCE

  # All results of models trained on the test data stored under "test_results"
    sapply(test_results, function(x) x)
  
  # Just create handpicked values for 24 features and predict cnt outcome based on model
    random_prediction <- matrix(c(0,6,14,0,1,2,0.30, 0.28, 0.66, 0.32, 340, 0,1,0,0,0,0,0,1,0,0,0))
    random_prediction <- t(random_prediction) # transpose column into matrix
  
  
  # PREDICTION ON TEST DATA
    test_predictions <- predict(dnn_model, as.matrix(test_features))
    test_predictions2 <- predict(dnn_model, as.matrix(random_prediction))
  
    summary(test_predictions2)  
  
  # Plot prediction 
  ggplot(data.frame(pred = as.numeric(test_predictions), cnt = test_labels$cnt)) +
    geom_point(aes(x = pred, y = cnt)) +
    geom_abline(intercept = 0, slope = 1, color = "blue")
  
  # Error distribution 
  qplot(test_predictions - test_labels$cnt, geom = "density")
  
  
  # Store Model
  save_model_tf(dnn_model, 'dnn_model')


# CROSS-VALIDATION

  train_control <- trainControl(method = "cv", number = 5)
  
  dnn_caret <- train(
    cnt  ~ .,
    data = dt_training[,-"id"],
    method = "neuralnet",
    layer1= 16, layer2=16,layer3=16,
    trControl = train_control)
  
  summary(dnn_caret)
  
  dnn_caret$results

# CROSS-VALIDATION BY LOOP (doesnt work yet)

  # K-fold Cross Validation
    K <- 5
  
  # Collect MSE across folds
    dt_cv_results <- data.table(fold=numeric(K), mse_cv=numeric(K))
  
  # Randomly assign data points to K folds
    dt_training <- split_K(data_training, K)
    
  # Set up model without normalization layer
    build_and_compile_model <- function(norm) {
      model <- keras_model_sequential() %>%
        layer_dense(16, activation = 'relu') %>% # DNN with 3 layers 
        layer_dense(16, activation = 'relu') %>% # and width of 16 neurons
        layer_dense(16, activation = 'relu') %>%
        layer_dense(1)
      
      model %>% compile(
        loss = 'mean_absolute_error',           # Attention: absolute loss here (alt. squared)
        optimizer = optimizer_adam(0.1)         # Learning rate of the GD
      )
      
      model
    }
  
  ### Alternative standardization to keras layer_normalization:
    
    # Creates and train the model for this fold
    dnn_cv_model <- build_and_compile_model()
    
  # Normalizes the features using standardization (z-score normalization)
for (i in 1:K) {
  # Extracts the training and validation data for the current fold
  training_fold <- dt_training[dt_training$id != i, ]
  validation_fold <- dt_training[dt_training$id == i, ]
  
  # Normalizes the features using standardization (z-score normalization)
  train_features <- scale(training_fold[, -"cnt"])
  train_labels <- training_fold$cnt
  validation_features <- scale(validation_fold[, -"cnt"])
  validation_labels <- validation_fold$cnt

  # Trains the model with training data
  cv_history <- dnn_cv_model %>% fit(
    as.matrix(train_features),    # only temp as example input to reduce computation
    as.matrix(train_labels),
    verbose = 0,
    epochs = 10
  )
  
  # Calculates the MSE on the validation fold using the trained model
  mse_cv <- mean((validation_labels - predict(dnn_cv_model, as.matrix(validation_features)))^2)
  
  # Stores the MSE and fold number in the results data.table
  dt_cv_results$mse_cv[i] <- mse_cv
  dt_cv_results$fold[i] <- i
}
    dt_cv_results[,  mean(mse_cv), by="fold"]
    
     summary(dt_cv_results)
    dt_cv_results    
     
### END 

    #TODO - maybe like this?
    
    # K-fold Cross Validation
    K <- 5
    
    # Collect MSE across folds
    dt_results <- data.table(fold=numeric(K), mse_cv=numeric(K))
    
    # Randomly assign data points to K folds
    dt_training <- split_K(dt_training, K)
    
    for (i in 1:K) {
      # Preprocessing goes here!
      dt_training_scaled <- data.table(scale(dt_training[id != i,]))
      dt_validation_scaled <- data.table(scale(dt_training[id == i,]))
      
      # Modelling 
       dnn_cv_model <- build_and_compile_model()
  
  # Trains the model with training data
  cv_history <- dnn_cv_model %>% fit(
    as.matrix(train_features),    # only temp as example input to reduce computation
    as.matrix(train_labels),
    verbose = 0,
    epochs = 10
  )
      
      # CV Error
      dt_results$mse_cv[i] <- mean((dt_validation_scaled[, RetailPrice]-
                                      predict(dnn_cv_model, newdata = dt_validation_scaled))^2)
      dt_results$fold[i] <- i
    }
    
    
    
    
    
     
    
# Validating main DNN model on different folds (but not retraining on each)
  for (i in 1:K) {
    # Extract the validation data for the current fold
    validation_fold <- dt_training[id == i, ]
    validation_features <- validation_fold %>% select(-cnt)
    validation_labels <- validation_fold$cnt
    
    # Normalize the features using standardization (z-score normalization)
    validation_features <- scale(as.matrix(validation_features))
    
    # Calculate the MSE on the validation fold using the pre-trained model
    mse_cv <- mean((validation_labels - predict(dnn_model, validation_features))^2)
    
    # Store the MSE and fold number in the results data.table
    dt_results$mse_cv[i] <- mse_cv
    dt_results$fold[i] <- i
  }
  
    dt_results <- rbindlist(results)
    dt_results[,  mean(mse_cv), by="rep"]
