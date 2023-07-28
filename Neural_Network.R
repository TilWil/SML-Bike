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

# in case python doesnt run
library(reticulate)  # Just in case layer_norm function doesnt run
use_condaenv("r-reticulate", required = TRUE)
library(tensorflow)



# PREPARING THE DATA

# Define dependent and independent variables
  train_features <- data_training %>% select(-cnt) 
  train_labels <- data_training %>% select(cnt)  # dependent (label)
  
  train_features_norm <- data_training_norm %>% select(-cnt) 
  train_labels_norm <- data_training_norm %>% select(cnt)  # dependent (label)
  
  test_features <- data_test %>% select(-cnt)  # Normalized as it will be used on the cv performance check
  test_labels <- data_test %>% select(cnt)
  
  test_features_norm <- data_test_norm %>% select(-cnt)  # Normalized as it will be used on the cv performance check
  test_labels_norm <- data_test_norm %>% select(cnt)

# Normalize to avoid scale issues first on train and then temp-sub dataset
  normalizer <- layer_normalization()  
  
  normalizer %>% adapt(as.matrix(train_features)) 
  
  temp_normalizer <- layer_normalization() # set normalization layer
  temp_normalizer %>% adapt(temp)  # adapt feature matrix to layer
  
 
############################## DEEP NEURAL NETWORKS#############################
  # 1. Visualize Deep Neural Network
  # 2. Multi feature input model (MAIN MODEL)
  # 3. Varying learning rate over single feature input model
  # 4. Re-run model on PCA Data



    
        ########################### MAIN MODEL ##############################
  
# 1. Visualize with neuralnet function
  # First define Neural Network
  dnn_viz <- neuralnet(cnt ~ yr+mnth+hr+holiday+weathersit+temp+atemp+hum+windspeed
                       +lagged_cnt+Spring+Summer+Fall+Winter+Sunday+Monday+Tuesday+
                         Wednesday+Thursday+Friday+Saturday,
                       data = data_training,
                       hidden=c(16,16), # neurons in the hidden layers
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
  
  # 2. Setting up the model with Keras
  build_and_compile_model <- function(norm) {
    model <- keras_model_sequential() %>%
      norm() %>%
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
  
  test_results <- list()
  
  
  test_results[['dnn_model']] <- dnn_model %>% evaluate(
    as.matrix(test_features),
    as.matrix(test_labels),
    verbose = 0)
  
  # On normalized test
  test_results[['dnn_model_norm']] <- dnn_model %>% evaluate(
    as.matrix(test_features_norm),
    as.matrix(test_labels_norm),
    verbose = 0)
  
  # Store Model
  save_model_tf(dnn_model, 'dnn_model')
  

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
  # pca_training_sample <- data_PC18[,sample(.N, floor(.N*.80))]
  # pca_data_training <- data_PC18[pca_training_sample]
  # pca_data_test <- data_PC18[-pca_training_sample]
  
  pca_train_features <- data_PC18 %>% select(-cnt)
  pca_train_labels <- data_PC18 %>% select(cnt)
  
  # pca_test_features <- pca_data_test %>% select(-cnt)
  # pca_test_labels <- pca_data_test %>% select(cnt)

# Same procedure with other data and no normalization layer as pca data already scaled
  build_and_compile_model_pca <- function(norm) {
    model <- keras_model_sequential() %>%
      layer_dense(16, activation = 'sigmoid') %>% # DNN with 3 layers 
      layer_dense(16, activation = 'sigmoid') %>% # and width of 16 neurons
      layer_dense(1)
    
    model %>% compile(
      loss = 'mean_squared_error',           # Attention: absolute loss here (alt. squared)
      metrics = "mae",
      optimizer = optimizer_adam(0.001)         # Learning rate of the GD
    )
    
    model
  }
  
  dnn_pca_model <- build_and_compile_model_pca()   # store new model with same model-set up 
  # but other Input Matrix
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

  test_results[['dnn_pca_model']] <- dnn_pca_model %>% evaluate(
    as.matrix(test_features),
    as.matrix(test_labels),
    verbose = 0
  )


     ########################### CROSS-VALIDATION ########################  

  # OPTION 1: Using Keras Validation Split function
  dnn_cv_model <- build_and_compile_model(normalizer)   # store new model with same model-set up 
  
  # Fitting the model
  dnn_cv_history <- dnn_cv_model %>% fit(        # Run model with fit command
    as.matrix(train_features),
    as.matrix(train_labels),
    validation_split = 0.2,
    shuffle = TRUE,  # If true training data is re-shuffeld before each epoch
    verbose = 0,
    epochs = 100
  )
  
  plot(dnn_cv_history)
  
  test_results[['dnn_cv_model']] <- dnn_cv_model %>% evaluate(
    as.matrix(test_features),
    as.matrix(test_labels),
    verbose = 0)
  
  save_model_tf(dnn_cv_model, 'dnn_cv_model')
  

# OPTION 2: CROSS-VALIDATION BY LOOP 

    # K-fold Cross Validation
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
    # Preprocessing goes here!
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
                               rep=100)  # needs to be one for visualization otherwise prints all reps.
    
  # Calculates the MSE on the validation fold using the trained model
    test_predictions <- predict(dnn_cv_model2, as.matrix(validation_features)) # Error from predict - only NA's
    mse_cv <- mean((validation_labels - test_predictions)^2) # dnn_cv_model
    
   #dt_results$mse_cv[i] <- mean((validation_labels - predict(dnn_cv_model, validation_features))^2)

  # Stores the MSE and fold number in the results data.table
    dt_cv_results$mse_cv[i] <- mse_cv
    dt_cv_results$fold[i] <- i
  }
    plot(cv_history)
    show(dt_cv_results)
    
    
  # Option 3: CV within Keras and scikit-learn KFold function but scikit-learn doesnt run
  # Option 4: Caret package build in neuralnet function - but computation >30min so option dropped

    
    
# PERFORMANCE

  # All results of models trained on the test data stored under "test_results"
    sapply(test_results, function(x) x)

  
  # Expected loss 
    #  Model dnn
    dnn_test_predictions <- predict(dnn_model, as.matrix(test_features))
    Expected_loss_dnn <- mean((as.matrix(test_labels) - dnn_test_predictions)^2)
    
    #  Model dnn_cv
    dnn_cv_test_predictions <- predict(dnn_cv_model, as.matrix(test_features))
    Expected_loss_dnn_cv <- mean((as.matrix(test_labels) - dnn_cv_test_predictions)^2)
    
    #  Model dnn_pca on non-normalized test data
    dnn_pca_test_predictions <- predict(dnn_pca_model, as.matrix(test_features))
    Expected_loss_dnn_pca <- mean((as.matrix(test_labels) - dnn_pca_test_predictions)^2)
    
    # Expected loss of pca model in pca space - not same test data as other models but unseen from model
    dnn_pca_test_predictions2 <- predict(dnn_pca_model, as.matrix(pca_test_features))
    Expected_loss_dnn_pca_on_pca <- mean((as.matrix(pca_test_labels) - dnn_pca_test_predictions2)^2) 
  
    # CV Model 2
    Expected_loss_dnn_cv2 <- mean(as.matrix((test_labels_norm - predict(dnn_cv_model2, test_features_norm))^2))
    
    # CV Model 2 no scale
    Expected_loss_dnn_cv2_no_scale <- mean(as.matrix((test_labels_norm - predict(dnn_cv_model2_no_scale, test_features_norm))^2))

### END 
 