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

source("Data preparation.R")
source("Linear_model.R")


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


# NEURAL NETWORK

# Generate data
f <- function(x){x*(x-1)*(sin(13*x)+cos(23*x)*(1-x))}
x <- scale(dt_nn)
y <- f(x)
draw <- sample(1:17379,30)
x_sample <- scale(dt_nn)[draw]
y_sample <- scale(data$cnt)[draw]

# Learning parameters
neurons <- 5
learning_rate <- 0.1
epochs <- 5000

# Set up data
X    <-  dt_nn
N    <-  length(x_sample)
loss <-  data.frame(loss=numeric(epochs), epoch = numeric(epochs))

# Initialize
Mu     <- c(runif(neurons)) 
Sigma  <- c(runif(neurons)) 
B_LOut <- matrix(c(runif(neurons)), nrow=25, ncol = neurons)
b_LOut <- c(runif(25)) 

for (i in 1:epochs){
  
  # Forward pass
  xb <- (X-Mu) # for use in the backward pass
  XB <- (xb^2) # for use the backward pass
  A_L1 <- exp((-1/(Sigma)^2)*XB)
  A_LOut <- (B_LOut %*% A_L1) + b_LOut  
  eps <- t(y_sample)-A_LOut
  
  loss$loss[[i]] <- mean(eps^2)
  loss$epoch[[i]] <- i
  
  # Backward pass
  dXB_LOut    <- -2*eps
  dXB         <- (t(B_LOut) %*% dXB_LOut)*(A_L1)
  dXB_Mu      <- dXB*((1/(Sigma)^2)*xb) 
  dXB_Sigma   <- dXB*((1/(Sigma)^3)*XB)
  
  b_LOut_Grad <- rowMeans(dXB_LOut)
  B_LOut_Grad <- (dXB_LOut %*% t(A_L1)) / N
  Mu_Grad     <- (dXB_Mu %*% x_sample) / N
  Sigma_Grad  <- (dXB_Sigma %*% x_sample) / N
  
  b_LOut      <- b_LOut - (learning_rate * b_LOut_Grad)
  B_LOut      <- B_LOut - (learning_rate * B_LOut_Grad)
  Mu          <- Mu - as.vector(learning_rate * Mu_Grad)
  Sigma       <- Sigma - as.vector(learning_rate * Sigma_Grad)
}

ggplot(data=loss, aes(x=epoch, y=loss)) +
  geom_line() +
  labs(
    x = "epoch",
    y = "loss") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))
  
  # DOenst run yet --> TODO fix


# NEURAL NETWORK WITH KERAS
X <- scale(dt_nn)
y <- scale(data$cnt)

normalize %>% adapt(X)

model <- keras_model_sequential()

model %>% 
  
  # L1: Preprocessing
  normalize() %>%
  
  # L2: Hidden
  layer_dense(name="HiddenLayer",
              units = 8, 
              activation = 'relu') %>%
  
  # L3: Output
  layer_dense(name = "OutputLayer",
              units = 1)