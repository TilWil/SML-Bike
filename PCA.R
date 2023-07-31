# PCA

# Authors: Hong Le, Tilman von Samson
# 31.07.2023

# Content:
# 1. Data normalization and PC calculation
# 2. Visualization of PC's
# 3. Effects of PC Transformation on Data
# 4. Dimensionality Reduced Data Creation

source("Data preparation.R")

library(data.table)
library(ggplot2)
library(gganimate)
library(factoextra)
library(reshape2)

# 1. NORMALIZATION AND COVARIANCE MATRIX
  # Get standardized data
  dt_pca <- data.table(scale(train_features))
  
  # Decompose covariance matrix in principal components with prcomp command
  model_pca <- prcomp(dt_pca) 
  
  summary(model_pca)

# 2. VISUALIZATION OF PC RELEVANCE
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
    geom_vline(xintercept = 18, col = "red") +
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

# 3. PRINCIPAL COMPONENT TRANSFORMATION
# a) Inspect how PCs transform location of datapoints
# b) Inspect how reducing PCs impacts reconstruction of datapoints
# c) Determine the number of PCs that should be used

    # (Intermediate step for understanding the method)

    # a) Inspect how PCs transform data-point location
    # x <- first row of dataset (=first hour observed)
    round(as.matrix(dt_pca)[1,],2)
    
    # x*V  <- first row times eigenvector gives location of datapoint in new PC coordinate system
    round(as.matrix(dt_pca)[1,] %*% 
      as.matrix(model_pca$rotation),2)
    
    # x*V*V^T <- reconstructing the initial datapoint-location by multiplying with transposed eigenvector
    round(as.matrix(dt_pca)[1,] %*% 
            as.matrix(model_pca$rotation) %*% 
            t(as.matrix(model_pca$rotation)), 2) 
    
    # b) Loop over all PC's and show average loss of PC reduction on data-point accuracy
    results <- list()
    for (pcs in 1:22){
      data_reconstructed <- as.matrix(dt_pca) %*% 
        as.matrix(model_pca$rotation[,1:pcs]) %*% 
        t(as.matrix(model_pca$rotation[,1:pcs])) 
      
      results[[pcs]] <- data.table("PCs used" = pcs,    # Stores difference between reconstructed and original data 
                                   "MAE"= mean(colMeans(abs(as.matrix(dt_pca)-data_reconstructed)))) # in absolute terms
    }
    
    dt_reconstruction <- rbindlist(results)
    round(dt_reconstruction,2)

  # c) Summary:
  # Results show that last 5 PCs can be neglected without any loss
  # Also above shown that >90 % of variance is explained within 14 PCs


# 4. DIMENSIONALITY REDUCED DATA
# (Copy in Data preparation file )

  X <- as.matrix(dt_pca[,])
  V17_reduced <- model_pca$rotation[,1:17] 
  data_PC17 <- data.table(X %*% V17_reduced %*% t(V17_reduced))
  
  # Alternatively 90% accuracy with 14 PC
  X <- as.matrix(dt_pca[,])
  V14_reduced <- model_pca$rotation[,1:14] 
  data_PC14 <- data.table(X %*% V14_reduced %*% t(V14_reduced))

# End