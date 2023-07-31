# Data preparation
# Authors: Hong Le, Tilman von Samson
# 31.07.2023

# source("Packages.R")

rm(list = ls())

library(dplyr)
library(data.table)

# Read CSV file
data <- fread(file = "C:\\Users\\tilma\\Documents\\Uni\\Master Economics\\Machine Learning\\Statistical Machine Learning\\Data\\hour.csv", sep = ",")

# Display data format
str(data)           # all num or int except dteday which is chr

# First glimpse at first rows
head(data)

# Inspect for missing values and print result
missing_values <- sum(is.na(data))
print(missing_values)
data <- na.omit(data)

# Add lagged count 
data <- data %>%
  mutate(lagged_cnt = lag(cnt, default=0))

# DUMMYS

# Convert "season" to a factor with custom labels
data$season <- factor(data$season, levels = 1:4, labels = c("Spring", "Summer", "Fall", "Winter"))

# Convert "weekday" to a factor with custom labels and relevel the factor to start from Monday (1)
data$weekday <- factor(data$weekday, levels = 0:6, labels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
# data$weekday <- relevel(data$weekday, ref = "Monday")

# Create dummy variables for "season" and "weekday"
season_dummies <- model.matrix(~ season - 1, data = data)
weekday_dummies <- model.matrix(~ weekday - 1, data = data)


# Update column names with the desired names
colnames(season_dummies) <- levels(data$season)
colnames(weekday_dummies) <- levels(data$weekday)


# Combine the dummy variables 
data <- cbind(data, season_dummies, weekday_dummies)

# Remove the original variables
data <- subset(data, select = -c(dteday, season, weekday, instant, registered, casual))


# Convert all variables to numeric using lapply and as.numeric
data[] <- lapply(data, as.numeric)

# Create Test and Training dataset (80% of data) and test dataset (20% of data)
set.seed(123456)
training_sample <- data[,sample(.N, floor(.N*.80))]
data_training <- data[training_sample]
data_test <- data[-training_sample]

data_training_norm <- data.table(scale(data[training_sample]))
data_test_norm <- data.table(scale(data[-training_sample]))

# Define dependent and independent variables
train_features <- data_training %>% select(-cnt) 
train_labels <- data_training %>% select(cnt)  # dependent (label)

test_features <- data_test %>% select(-cnt) 
test_labels <- data_test %>% select(cnt)


################ DIMENSIONALITY REDUCED DATA #################
# Derivation of number of reduced PCs in PCA file 

# Get standardized data
dt_pca <- data.table(scale(train_features))

# Decompose covariance matrix in principal components with prcomp command
model_pca <- prcomp(dt_pca)

# Compute PC reduced dataset
X <- as.matrix(dt_pca[,])
V17_reduced <- model_pca$rotation[,1:17] 
data_PC17 <- data.table(X %*% V17_reduced %*% t(V17_reduced))

X <- as.matrix(dt_pca[,])
V14_reduced <- model_pca$rotation[,1:14] 
data_PC14 <- data.table(X %*% V14_reduced %*% t(V14_reduced))

# Only features to avoid information leakage on dependent
feature_pca <- data.table(scale(data_training[, -"cnt"]))  
model_feature_pca <- prcomp(feature_pca)   

X <- as.matrix(feature_pca[,])
V17_reduced <- model_feature_pca$rotation[,1:17] 
feature_PC17 <- data.table(X %*% V17_reduced %*% t(V17_reduced))



# END


