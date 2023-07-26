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




# NEURAL NETWORK WITH KERAS

# Define dependent and independent variables
train_features <- data_training %>% select(-cnt) 
train_labels <- data_training %>% select(cnt)  # dependent (label)

test_features <- data_test %>% select(-cnt)
test_labels <- data_test %>% select(cnt)

# Normalize to aviod scale issues first on train and then test dataset
normalizer <- layer_normalization(axis = -1)  # -1 so all features are normalized independently

normalizer %>% adapt(as.matrix(train_features)) # apply layer on feature space

test_normalizer <- layer_normalization(axis = -1)

test_normalizer %>% adapt(as.matrix(test_features))

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
  
  # Now as only two-dimensional model we can plot the reuslts
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


### DEEP NEURAL NETWORKS

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

### SINGLE INPUT DNN
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

# MULTI INPUT DNN
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

# DNN WITH VARYING LEARNING RATES

# APPROACH:
  # To inspect on varying settings of the learning rate on model performance we set-up 
  # a loop within the deep neural netweork is trained on the single "temp" input (to 
  # reduce computation only "temp") 
  # Therefore: 
  # 1. specify model with option of varying optimizer settings
  # 2. Loop different DNNs on the training data and saver results in "training_histories"
  # 3. Plot results in "training_histories" with a little plot loop 

# 1. Function to build and compile the model with a given normalization and learning rate
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

# 2. Loop through each learning rate and train the model
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


# 3. Plot all the training and validation losses together

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


# PERFORMANCE

# All results of models trained on the test data stored under "test_results"
sapply(test_results, function(x) x)

# Just create random variables for 24 features and predict cnt outcome based on model
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



# RERUN MODEL on PCA dimensionality reduced data
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



# compare loss reduction
par(mfrow = c(1, 2))  # Plotting layout - arranges 3 plots side by side

plot(pca_history)
plot(dnn_history)
par(mfrow = c(1, 1))  # Plotting layout - arranges 3 plots side by side

test_results[['dnn_pca_model']] <- dnn_pca_model %>% evaluate(
  as.matrix(pca_test_features),
  as.matrix(pca_test_labels),
  verbose = 0
)

# compare test results
sapply(test_results, function(x) x)



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

# CROSS-VALIDATION BY LOOP

# K-fold Cross Validation
K <- 5

# Collect MSE across folds
dt_results <- data.table(fold=numeric(K), mse_cv=numeric(K))

# Randomly assign data points to K folds
dt_training <- split_K(data_training, K)

### Alternative standardization to keras layer_normalization:

# Normalize the features using standardization (z-score normalization)

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

### END






























# Notes
# Preprocessing with keras doesnt work inthe loop:
for (i in 1:K) {
  # Preprocessing goes here!

  train_validation_features <- dt_training %>% select(-cnt)
  train_validation_labels <- dt_training %>% select(cnt)
  
  dt_validation_scaled <- layer_normalization(axis = -1)  # -1 so all features are normalized independently
  dt_validation_scaled %>% adapt(as.matrix(train_validation_features[id != i,])) # apply layer on feature sp
  
  # CV Error
  dt_results$mse_cv[i] <- mean((train_validation_labels -
                                  predict(dnn_model, newdata = train_validation_features))^2)
  dt_results$fold[i] <- i
}

# Does not work as tensorflow does not allow the normalization per loop

# Just for practice predict outcome when temp is 0.28 and hum 0.81

new_data <- data.frame(temp = 0.28, hum = 0.81)

x_new <- matrix(c(0,28), ncol = 1)
dnn_temp_model %>% predict(x=x_new,verbose=0)

random_values <- runif(25, min = -15, max = 15)
df <- data.frame(Random_Column = random_values)

predict <- dnn_model %>% predict(x=x_new,verbose=0)

predicted_values <- predict(dnn_model, as.matrix(x_new))
print(predicted_values)

# Following the regression tutorial steps from tensorflow
# inspect on data normalization
print(normalizer$mean)
first <- as.matrix(features[1,])
cat('First example:', first)
cat('Normalized:', as.matrix(normalizer(first)))


# Initialize Sigmoid (S) and Relu functions for activation of the neuron
S <- function(xb){ return(1/(1+exp(-xb))) }
dS <- function(S) { S*(1-S) }

RELU <- function(xb) { pmax(xb,0) }
dRELU <- function(R) { pmin(pmax(R,0),1) }


### GRADIENT DESCEND (for intuition)

# Question 5. setting of descend stepsize
learning_rate <- 0.1
epochs <- 100

y <- data$cnt
set.seed(12345)
# Data setup
X <- scale(dt_nn)
y <- matrix(y, ncol = 1)
N <- nrow(X)

# Plotting
loss <-  data.frame(loss=numeric(epochs), epoch = numeric(epochs), 
                    b0 = numeric(epochs), b1 = numeric(epochs))

# Initialize the first location of betas 
B <- matrix(c(runif(1)), ncol=16)
b <- c(runif(1)) 

for (i in 1:epochs){
  
  # Forward pass
  XB  <- (B %*% t(X)) + b 
  eps <- t(y) - XB
  
  # Backward pass
  b_Grad <- (-2/N) * sum(eps)
  B_Grad <- (-2/N) * (eps %*% X) 
  b   <- b - (learning_rate * b_Grad)
  B   <- B - (learning_rate * B_Grad)
  
  # Collecting results for plotting 
  loss$loss[[i]] <- sqrt(mean(eps^2))
  loss$epoch[[i]] <- i
  loss$b0[[i]] <- b
  loss$b2[[i]] <- B
  loss$b3[[i]] <- B
  loss$b4[[i]] <- B
  loss$b5[[i]] <- B
  loss$b6[[i]] <- B
  loss$b7[[i]] <- B
  loss$b8[[i]] <- B
  loss$b9[[i]] <- B
  loss$b10[[i]] <- B
  loss$b11[[i]] <- B
  loss$b12[[i]] <- B
  loss$b13[[i]] <- B
  loss$b14[[i]] <- B
  loss$b15[[i]] <- B
  loss$b16[[i]] <- B
  
  
}

# Plot the losses as a function of runs of forward and backward passes
ggplot(data=loss, aes(x=epoch, y=loss)) +
  geom_line() +
  labs(
    x = "epoch",
    y = "loss") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))


sqrt(mean(lm_full$residuals^2))
sqrt(mean(eps^2))
