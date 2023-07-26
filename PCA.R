# PCA
source("Data preparation.R")

library(data.table)
library(ggplot2)
library(gganimate)
library(factoextra)
library(reshape2)

# Get standardized data
dt_pca <- data.table(scale(data))

# Decompose covariance matrix in principal components with prcomp command
model_pca <- prcomp(dt_pca) 

summary(model_pca)


fviz_eig(model_pca, addlabels = TRUE, ncp=23) +    # inspect on the 23 PC's
  theme(plot.title = element_text(hjust = 0.5))

# Plot cumulative variance explained by the PC's 
df_cumsum <- data.table(Var = model_pca$sdev^2 / sum(model_pca$sdev^2), PC = 1:ncol(dt_pca))

ggplot(data=df_cumsum, aes(y=cumsum(Var), x=PC)) +
  geom_line() +
  labs(
    title = "Cumulative Variance Explained",
    x = "Principal Component",
    y = "% Variance Explained") +
  geom_line(y = 0.9, col = "red") +
  scale_x_continuous(breaks=c(1:nrow(df_cumsum))) +
  theme_minimal() +
  theme(axis.line = element_blank(),
        plot.title = element_text(hjust = 0.5))

# How much original variables contribute to PC's
round(model_pca$rotation,2)

fviz_pca_var(model_pca, axes = c(1,2),
             col.var="contrib",
             gradient.cols = c("white", "blue", "red"),
             ggtheme = theme_minimal(),
             title = "Loadings Plot",
             repel = TRUE) +
  theme_minimal()

# COORDINATE TRANSFORMATION
# First by hand then automated
# 1. Inspect how PCs transform location of datapoints
# 2. Inspect how reducing PCs impacts reconstruction of datapoints
# 3. Determine the number of PCs that should be used


    ########### Intermediate steps for understanding the process ###############

    # 1. BY HAND
    # x <- first row of dataset (=first hour observed)
    round(as.matrix(dt_pca)[1,],2)
    
    # x*V  <- first row times eigenvector gives location of datapoint in new PC coordinate system
    round(as.matrix(dt_pca)[1,] %*% 
      as.matrix(model_pca$rotation),2)
    
    # x*V*V^T <- reconstructing the initial datapoint-location by multiplying with transposed eigenvector
    round(as.matrix(dt_pca)[1,] %*% 
            as.matrix(model_pca$rotation) %*% 
            t(as.matrix(model_pca$rotation)), 2) 
    
    # Now reduce PCs and inspect how accuracy of reconstruction is impacted
    
    # x * V_reduced * V_reducedT
    PC3 <- as.matrix(dt_pca)[1,] %*% 
      as.matrix(model_pca$rotation[,1:3]) %*% 
      t(as.matrix(model_pca$rotation[,1:3]))
    
    
    PC14 <- as.matrix(dt_pca)[1,] %*% 
      as.matrix(model_pca$rotation[,1:14]) %*%    # 90% of variance explained
      t(as.matrix(model_pca$rotation[,1:14]))
    
    PC23 <- as.matrix(dt_pca)[1,] %*% 
      as.matrix(model_pca$rotation[,1:23]) %*% 
      t(as.matrix(model_pca$rotation[,1:23]))
    
    PC23Loss  <- sum(as.matrix(dt_pca)[1,] -PC23) # Difference to original observation
    PC14Loss  <- sum(as.matrix(dt_pca)[1,] -PC14) 
    PC3Loss   <- sum(as.matrix(dt_pca)[1,] -PC3)
    
    par(mfrow = c(1, 3))  # Plotting layout - arranges 3 plots side by side
    plot(PC3Loss)
    plot(PC14Loss)
    plot(PC23Loss)        # Mean absolute error becomes 0 
    
    
    # Loop over all PC's and show average loss on the datapoint
    results <- list()
    
    for (pcs in 1:23){
      data_reconstructed <- as.matrix(dt_pca)[1,] %*% 
        as.matrix(model_pca$rotation[,1:pcs]) %*% 
        t(as.matrix(model_pca$rotation[,1:pcs])) 
      
      results[[pcs]] <- data.table("PCs used" = pcs,
                                   "MAE"= sum(abs(as.matrix(dt_pca)[1,]-data_reconstructed)),
                                   data_reconstructed)
    }
    
    dt_reconstruction <- rbindlist(results)
    round(dt_reconstruction,2)
    
    results <- list()

############################## Main Approach ###################################
    
# Now on the whole dataset
for (pcs in 1:23){
  data_reconstructed <- as.matrix(dt_pca) %*% 
    as.matrix(model_pca$rotation[,1:pcs]) %*% 
    t(as.matrix(model_pca$rotation[,1:pcs])) 
  
  results[[pcs]] <- data.table("PCs used" = pcs,
                               "MAE"= mean(colMeans(abs(as.matrix(dt_pca)-data_reconstructed))))
}

dt_reconstruction <- rbindlist(results)
round(dt_reconstruction,2)

# Summary:
# Results show that last 5 PCs can be neglected without any loss
# Also above shown that >90 % of variance is explained within 14 PCs


################ DIMENSIONALITY REDUCED DATA #################

X <- as.matrix(dt_pca[,])
V18_reduced <- model_pca$rotation[,1:18] 
data_PC18 <- data.table(X %*% V18_reduced %*% t(V18_reduced))

X <- as.matrix(dt_pca[,])
V14_reduced <- model_pca$rotation[,1:14] 
data_PC14 <- data.table(X %*% V14_reduced %*% t(V14_reduced))

# End