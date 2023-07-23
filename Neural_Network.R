# Neural Network

# ! Not Functioning yet

# Questions in advance:
#   1. How much input?
#   2. How many Hidden layers?
#   3. Width of the layer? (how many neurons?)
#   4. What kind of activation function? (perceptron: logistic? tahn? Relu?)
#   5. Stepsize of the gradient


library(data.table)
library(ggplot2)
library(keras)
library(plot3D)
library(viridisLite)
library(tensorflow)
library(reticulate)
library(tidyverse)
library(tidymodels)

source("Data preparation.R")
source("Linear_model.R")


# NEURAL NETWORK WITH KERAS
features <- data %>% select(-cnt)
labels <- data %>% select(cnt)

normalizer <- layer_normalization(axis = -1L)

normalizer %>% adapt(as.matrix(features))

model <- keras_model_sequential()

# Following the regression tutorial steps from tensorflow
# inspect on data normalization
print(normalizer$mean)
first <- as.matrix(features[1,])
cat('First example:', first)
cat('Normalized:', as.matrix(normalizer(first)))

# One-feature linear regression - predicting cnt by temp
# First create normalized feature matrix
temp <- matrix(features$temp)  # create matrix
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

# Keras fit() function will now estimate 1oo epochs
history <- temp_model %>% fit(
  as.matrix(features$temp),
  as.matrix(labels),
  epochs = 100,
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


# Multi-variable linear regression
linear_model <- keras_model_sequential() %>%
  normalizer() %>%
  layer_dense(units = 1)

predict(linear_model, as.matrix(features[1:10, ]))

linear_model$layers[[2]]$kernel

linear_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.1),
  loss = 'mean_squared_error'
)

# Similar to above only with whole feature-space
history <- linear_model %>% fit(
  as.matrix(features),
  as.matrix(labels),
  epochs = 100,
  # Suppress logging.
  verbose = 0,
  # Calculate validation results on 20% of the training data.
  validation_split = 0.2
)

# Plot loss history
plot(history)


# Deep Neural Network with 3 layers and width of 16 neurons
build_and_compile_model <- function(norm) {
  model <- keras_model_sequential() %>%
    norm() %>%
    layer_dense(16, activation = 'relu') %>%
    layer_dense(16, activation = 'relu') %>%
    layer_dense(16, activation = 'relu') %>%
    layer_dense(1)
  
  model %>% compile(
    loss = 'mean_absolute_error',
    optimizer = optimizer_adam(0.001)
  )
  
  model
}

# first with single input
dnn_temp_model <- build_and_compile_model(temp_normalizer)

summary(dnn_temp_model)

history <- dnn_temp_model %>% fit(
  as.matrix(features$temp),
  as.matrix(labels),
  validation_split = 0.2,
  verbose = 0,
  epochs = 100
)

plot(history)

# As its only 2-D it's easy to plot the prediction
x <- seq(0.0, 1, length.out = 1.01)
y <- predict(dnn_temp_model, x)

ggplot(data) +
  geom_point(aes(x = temp, y = cnt, color = "data")) +
  geom_line(data = data.frame(x, y), aes(x = x, y = y, color = "prediction"))

# Same approach with multi-input feature-space
dnn_model <- build_and_compile_model(normalizer)

history <- dnn_model %>% fit(
  as.matrix(features),
  as.matrix(labels),
  validation_split = 0.2,
  verbose = 0,
  epochs = 100
)

plot(history)


# Until here works well, have to document the steps


# Just for practice predict outcome when temp is 0.28 and hum 0.81

new_data <- data.frame(temp = 0.28, hum = 0.81)

normalizer_nd <- layer_normalization(axis = -1L)

normalizer_nd %>% adapt(as.matrix(new_data))

predicted_values <- predict(dnn_model, as.matrix(new_data))
print(predicted_values)








# Notes

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
