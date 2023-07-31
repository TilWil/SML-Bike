# Neural Network
# Authors: Hong Le, Tilman von Samson
# 31.07.2023

# Content:
# 1. Data Preparation
# 2. Visualization of the Neural Network
# 3. NN Setup and Estimation
# 4. Hyperparameter adjustments
# 5. Rerun of NN on PCA Data
# 6. Cross-Validation
# 7. Perfomance on Test Data

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

# in case python doesnt run
library(reticulate)  # Just in case layer_norm function doesnt run
use_condaenv("r-reticulate", required = TRUE)
library(tensorflow)



# 1. PREPARING THE DATA

# Normalize to avoid scale issues first on train and then temp-sub dataset
  normalizer <- layer_normalization()  
  
  normalizer %>% adapt(as.matrix(train_features)) 
  
 
############################# DEEP NEURAL NETWORKS #############################
  # 2. Visualize Deep Neural Network
  # 3. Multi feature input model (MAIN MODEL)
  # 4. Hyperparameter settings: Varying learning rate and epochs
  # 5. Re-run model on PCA Data

    
        ########################### MAIN MODEL ##############################
  
# 2. Visualize with neuralnet function
  # First define Neural Network
  dnn_viz <- neuralnet(cnt ~ yr+mnth+hr+holiday+weathersit+temp+atemp+hum+windspeed
                       +lagged_cnt+Spring+Summer+Fall+Winter+Sunday+Monday+Tuesday+
                         Wednesday+Thursday+Friday+Saturday,
                       data = data_training,
                       hidden=c(16,16,16), # neurons in the hidden layers
                       learningrate= 0.1,
                       algorithm = "backprop",
                       act.fct = "logistic",
                       err.fct = "sse",   # sum of squared error as error function
                       linear.output = FALSE,
                       lifesign = "full", # Print outputs while running
                       rep=1)  # needs to be one for visualization otherwise prints all reps.
  # Then plot
  plot(dnn_viz, fill="green", col.hidden.synapse="lightblue",
       show.weights=FALSE,
       information=FALSE)

# 3. Setting up the model with Keras 
  build_and_compile_model <- function(norm) {     # Function that defines Network architecture and optimization parameter
    model <- keras_model_sequential() %>%
      norm() %>%                                  # Normalization Layer enters here
      layer_dense(16, activation = 'sigmoid') %>% # DNN with 3 layers 
      layer_dense(16, activation = 'sigmoid') %>% # and width of 16 neurons
      layer_dense(16, activation = 'sigmoid') %>%
      layer_dense(1)
    
    model %>% compile(
      loss = 'mean_squared_error',           # Attention: absolute loss here (alt. squared)
      metrics = "mae",              # Absolute error her also interesting: accuracy (Calculates how often predictions equal labels) 
      optimizer = optimizer_adam(0.001)         # Learning rate of the GD
    )
    
    model
  }
  # Use the compiled model from above
  dnn_model <- build_and_compile_model(normalizer)   # store new model with same model-set up 
  #summary(dnn_model)                                             
  
  # Fitting the model
  dnn_history <- dnn_model %>% fit(        # Run model with fit command
    as.matrix(train_features),
    as.matrix(train_labels),
    #validation_split = 0.2,
    verbose = 0,
    epochs = 100
  )
  
  plot(dnn_history)
  
  # Store Model
  save_model_tf(dnn_model, 'dnn_model')
  

# 4. HYPERPARAMETER

# APPROACH:
  # To inspect on varying settings of the learning rate and epochs on model  
  # performance we set-up a loop over the deep neural network 
  # Therefore: 
  # a. specify model with option of varying optimizer settings
  # b. Loop different DNNs on the training data and saver results in "training_histories"
  # c. Plot results in "training_histories" with a little plot loop
  # d. Apply similar loop to number of epochs

  # 4.a) Function to build and compile the model with a given normalization and learning rate
  build_and_compile_model_lr <- function(norm, learning_rate) {
    model <- keras_model_sequential() %>%
      norm() %>%
      layer_dense(16, activation = 'sigmoid') %>%
      layer_dense(16, activation = 'sigmoid') %>%
      layer_dense(1)
    
    model %>% compile(
      loss = 'mean_squared_error',
      optimizer = optimizer_adam(learning_rate) # Set the learning rate here
    )
    
    model
  }

  # List of learning rates 
  learning_rate <- c(0.1, 0.01, 0.001)

  # List to store training histories
  training_histories <- list()
  
  # 4.b) Loop through each learning rate and train the model
  for (lr in learning_rate) {
    # Create the model with the specified learning rate
    dnn_var_model <- build_and_compile_model_lr(normalizer, lr)
    
    # Train the model with training data
    history <- dnn_var_model %>% fit(
      as.matrix(train_features),    # only temp as example input to reduce computation
      as.matrix(train_labels),
      #validation_split = 0.2,
      verbose = 0,
      epochs = 50
    )
    
    # Store the training history in the list
    training_histories[[as.character(lr)]] <- history
  }


  # 4.c) Plot all the training and validation losses together

  # Save the plot as a PDF file
    pdf("training_rates_plots2.pdf")
  
    par(mfrow = c(1, length(learning_rate)))  # Plotting layout - arranges 3 plots side by side
    for (i in seq_along(learning_rate)) {    # loop that plots model-histories according to learning rate
      lr <- learning_rate[i]
      history <- training_histories[[as.character(lr)]]
      
      plot(history$metrics$loss, type = "l", col = "blue", main = paste("Learning Rate =", lr))
      lines(history$metrics$val_loss, col = "red")
      legend("topright", c("Training Loss", "Validation Loss"), col = c("blue", "red"), lty = 1)
    }
    par(mfrow = c(1, 1))  # Reset the plotting layout to default
  
  # close pdf file
  dev.off()

  
  # 4.d) Increasing the number of epochs
  epochs <- c(50, 100, 500)
  
  # store training histories
  training_histories_ep <- list()
  
  # Loop through each epoch and train the model
  for (ep in epochs) {
    # Create the model with the specified learning rate
    dnn_var_model <- build_and_compile_model(normalizer)
    
    # Train the model with training data
    history <- dnn_var_model %>% fit(
      as.matrix(train_features),    # Use appropriate training features
      as.matrix(train_labels),
      verbose = 0,
      epochs = ep
    )
    
    # Store the training history in the list
    training_histories_ep[[as.character(ep)]] <- history
  }
  
  # Save the plot as a PDF file
  pdf("epochs_plots2.pdf")
  
  par(mfrow = c(1, length(epochs)))  # Plotting layout - arranges 3 plots side by side
  for (i in seq_along(epochs)) {    # loop that plots model-histories according to learning rate
    ep <- epochs[i]
    history <- training_histories_ep[[as.character(ep)]]
    
    plot(history$metrics$loss, type = "l", col = "blue", main = paste("Learning Rate =", ep))
    lines(history$metrics$val_loss, col = "red")
    legend("topright", c("Training Loss", "Validation Loss"), col = c("blue", "red"), lty = 1)
  }
  par(mfrow = c(1, 1))  # Reset the plotting layout to default
  
  # close pdf file
  dev.off()
  
  
# 5. RERUN MODEL ON PCA DATA
  pca_train_features <- feature_PC18  # Rename just for better oversight 
  pca_train_labels <- train_labels_norm
  

  # Same procedure with other data and no normalization layer as pca data is 
  # already scaled
  build_and_compile_model_pca <- function(norm) {
    model <- keras_model_sequential() %>%
      layer_dense(16, activation = 'sigmoid') %>% # DNN with 3 layers 
      layer_dense(16, activation = 'sigmoid') %>% # and width of 16 neurons
      layer_dense(16, activation = 'sigmoid') %>% 
      layer_dense(1)
    
    model %>% compile(
      loss = 'mean_squared_error',           # Attention: absolute loss here (alt. squared)
      metrics = "mae",
      optimizer = optimizer_adam(0.001)         # Learning rate of the GD
    )
    
    model
  }
  
  dnn_pca_model <- build_and_compile_model_pca()   # store new model with model-set up from above 
  
  pca_history <- dnn_pca_model %>% fit(        # Run model with fit command
    as.matrix(pca_train_features),
    as.matrix(pca_train_labels),
    # validation_split = 0.2,
    verbose = 0,
    epochs = 100)
  
  # Store Model
  save_model_tf(dnn_pca_model, 'dnn_pca_model')

# Compare loss reduction
  plot(pca_history)
  plot(dnn_history)


# 6.CROSS-VALIDATION

# OPTION 1: Using Keras Validation Split function
  dnn_cv_model <- build_and_compile_model(normalizer)   # store new model with same model-set up 
  
  # Fitting the model
  dnn_cv_history <- dnn_cv_model %>% fit(        # Run model with fit command
    as.matrix(train_features),
    as.matrix(train_labels),
    validation_split = 0.2,
    shuffle = TRUE,  # If true training data is re-shuffeld before each epoch
    verbose = 0,
    epochs = 100 )
  
  plot(dnn_cv_history)
  
  save_model_tf(dnn_cv_model, 'dnn_cv_model')
  
  # Same with PCA data
  dnn_cv_pca_model <- build_and_compile_model_pca(normalizer)   # store new model with same model-set up 
  
  dnn_cv_pca_history <- dnn_cv_pca_model %>% fit(        # Run model with fit command
    as.matrix(pca_train_features),
    as.matrix(pca_train_labels),
    validation_split = 0.2,
    shuffle = TRUE,  # If true training data is re-shuffeld before each epoch
    verbose = 0,
    epochs = 100
  )
  
  plot(dnn_cv_pca_history)
  
  save_model_tf(dnn_cv_pca_model, 'dnn_cv_pca_model')
  

# OPTION 2: K-Fold CROSS-VALIDATION BY LOOP 

  # K-folds
    K <- 5
  
  # Collect MSE across folds
    dt_cv_results <- data.table(fold=numeric(K), mse_cv=numeric(K), residuals=numeric(K))
  
  # Randomly assign data points to K folds
    split_K <- function(data, folds){
      data$id <- ceiling(sample(1:nrow(data), replace = FALSE, nrow(data)) / # Create an id variable
                           (nrow(data) / folds))         # which randomly assigns each row to a fold 
      return(data)
    }
    # Randomly assign data points to K folds
    dt_training <- split_K(data_training, K)
    
    for (i in 1:K) {
    # Scaling within the Loop to avoid information leakage (mean and sd of "unseen" folds)
    dt_training_scaled <- data.table(scale(dt_training[id != i,]))
    dt_validation_scaled <- data.table(scale(dt_training[id == i,]))
    
    # define features and labels
    train_labels <- dt_training_scaled$cnt
    train_features <- dt_training_scaled[, -"cnt"]
    
    validation_features <- dt_validation_scaled[, -"cnt"]
    validation_labels <- dt_validation_scaled$cnt
  
    # Trains the model with new training data each round
    dnn_cv_model2 <- neuralnet(cnt ~ yr+mnth+hr+holiday+weathersit+temp+atemp+hum+windspeed+
                               lagged_cnt+Spring+Summer+Fall+Winter+Sunday+Monday+Tuesday+
                               Wednesday+Thursday+Friday+Saturday,
                               data = dt_training_scaled,
                               hidden=c(16,16,16), # neurons in the hidden layers
                               learningrate= 0.05,
                               algorithm = "backprop",
                               act.fct = "logistic",
                               err.fct = "sse",   # sum of squared error as error function
                               linear.output = FALSE,
                               lifesign = "full", # Print outputs while running
                               rep=100)  
    
  # Calculates the MSE on the validation fold using the trained model
    test_predictions <- predict(dnn_cv_model2, as.matrix(validation_features)) # Error from predict - only NA's
    mse_cv <- mean((validation_labels - test_predictions)^2) # dnn_cv_model
    
   #dt_results$mse_cv[i] <- mean((validation_labels - predict(dnn_cv_model, validation_features))^2)

  # Stores the MSE and fold number in the results data.table
    dt_cv_results$mse_cv[i] <- mse_cv
    dt_cv_results$fold[i] <- i
  }
    show(dt_cv_results)
    
    
  # Option 3: CV within Keras and scikit-learn KFold function but scikit-learn doesnt run
  # Option 4: Caret package build in neuralnet function - but computation >30min so option dropped

    
    
# 7. PEFROMANCE ON TEST DATA
  
  # Expected loss 
    #  Model dnn
    dnn_test_predictions <- predict(dnn_model, as.matrix(test_features))
    Expected_loss_dnn <- mean((as.matrix(test_labels) - dnn_test_predictions)^2)
    
    #  Model dnn_cv
    dnn_cv_test_predictions <- predict(dnn_cv_model, as.matrix(test_features))
    Expected_loss_dnn_cv <- mean((as.matrix(test_labels) - dnn_cv_test_predictions)^2)
    #  Model dnn_cv
    dnn_cv_pca_test_predictions <- predict(dnn_cv_pca_model, as.matrix(test_features))
    Expected_loss_dnn_cv_pca <- mean((as.matrix(test_labels) - dnn_cv_test_predictions)^2)
    
    #  Model dnn_pca 
    dnn_pca_test_predictions <- predict(dnn_pca_model, as.matrix(test_features))
    Expected_loss_dnn_pca <- mean((as.matrix(test_labels) - dnn_pca_test_predictions)^2)
  
    # CV Model 2
    Expected_loss_dnn_cv2 <- mean(as.matrix((test_labels - predict(dnn_cv_model2, test_features))^2))

### END 
 