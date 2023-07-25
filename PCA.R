# PCA
source("Data preparation.R")

library(data.table)
library(ggplot2)
library(gganimate)
library(factoextra)
library(reshape2)

# Get standardized data

dt_pca <- data.table(scale(data))

# decompose covariance matrix in principal comonents with prcomp

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

# BY HAND
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
# x * V_reduced
as.matrix(dt_pca)[1,] %*% 
  as.matrix(model_pca$rotation[,1:14]) 

# x * V_reduced * V_reducedT
PC3 <- as.matrix(dt_pca)[1,] %*% 
  as.matrix(model_pca$rotation[,1:3]) %*% 
  t(as.matrix(model_pca$rotation[,1:3]))


PC14 <- as.matrix(dt_pca)[1,] %*% 
  as.matrix(model_pca$rotation[,1:14]) %*% 
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
# Also >90 % of variance is explained within 14 PCs

X <- as.matrix(dt_pca[,])
V18_reduced <- model_pca$rotation[,1:18] 
data_PC18 <- data.table(X %*% V18_reduced %*% t(V18_reduced))

X <- as.matrix(dt_pca[,])
V14_reduced <- model_pca$rotation[,1:14] 
data_PC14 <- data.table(X %*% V14_reduced %*% t(V14_reduced))

# End


















# Notes

data_PC18 <- data.table(as.matrix(dt_pca) %*% 
  as.matrix(model_pca$rotation[,1:18]) %*% 
  t(as.matrix(model_pca$rotation[,1:18])))



reduced_pca <- data.table(as.matrix(dt_pca) %*% as.matrix(model_pca$rotation[, 1:18]) %*% 
  t(as.matrix(model_pca$rotation[, 1:18])))


num_components <- 14

data_PC14 <- data.frame(matrix(NA, nrow = nrow(dt_pca), ncol = num_components))

# Perform PCA for each row in dt_pca
for (i in 1:nrow(dt_pca)) {
  reduced_row <- as.matrix(dt_pca[i,]) %*% as.matrix(model_pca$rotation[, 1:num_components]) %*% t(as.matrix(model_pca$rotation[, 1:num_components]))
  data_PC14[i, ] <- reduced_row
}

num_components <- 14

# Apply PCA to all rows of dt_pca and store the result in data_PC14
data_PC14 <- t(apply(dt_pca, 1, function(row) {
  reduced_row <- as.matrix(row) %*% as.matrix(model_pca$rotation[, 1:num_components]) %*% t(as.matrix(model_pca$rotation[, 1:num_components]))
  return(reduced_row)
}))

# Convert the transposed result back to a dataframe
data_PC14 <- as.data.frame(data_PC14)





# Kernels
library(kernlab)

# PCA

dt_kpca <- data
model_kpca <- kpca(matrix(c(dt_kpca$hr,dt_kpca$temp), ncol = 2), kernel="rbfdot", kpar=list(sigma=1))


index <- c(rep(names(dt_pca)))

# Plot the differences
plot(index, PC3Loss, type = "l", col = "red", xlab = "Entry Index", ylab = "Differences", main = "Losses")
lines(index, PC14Loss, col = "blue")
lines(index, PC23Loss, col = "green")
legend("topright", legend = c("Vector1 - Vector2", "Vector1 - Vector3", "Vector2 - Vector3"), col = c("red", "blue", "green"), lty = 1)

# Plot results

PCLoss <- data.frame(Variable = rep(names(dt_pca), 3), Value = c(PC23Loss, PC14Loss, PC3Loss))
melted_data <- melt(PCLoss, id.vars = "Value") # Melt the data into long format
ggplot(melted_data, aes(x = Value)) +
  geom_histogram(binwidth = 0.05, aes(fill = variable), alpha = 0.7) +
  labs(title = "Histogram of Dimensionality-Reduction-Losses",
       x = "Value", y = "Frequency") +
  scale_fill_manual(values = c("green","red", "blue"))  # Specify colors for PC14Loss and PC3Loss
