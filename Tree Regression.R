library(rpart)
library(rpart.plot)

# Tree regression using all dependent variable except for lagged count
model_tree_nonlag <- rpart(data=data_training, formula = cnt ~ yr + mnth + hr +
                      holiday + workingday +
                      Spring + Summer + Fall + Winter +
                      Monday + Tuesday + Wednesday + Thursday +
                      Friday + Saturday + Sunday +
                      weathersit + temp + atemp + hum + windspeed,
                    method="anova")
prp(model_tree_nonlag)
printcp(model_tree_nonlag)

# Add lagged count
model_tree <- rpart(formula = cnt ~ ., data=data_training, method="anova")
prp(model_tree)

# Tabular and a graphical overview of the cp parameter, which captures model complexity
printcp(model_tree)
plotcp(model_tree)

# Choosing the cp value that minimizes the cross-validation error (xerror): the last row in the cp table
min_cp <- model_tree$cptable[which.min(model_tree$cptable[, "xerror"]), "CP"]
# Prune our model object
model_tree_prune <- prune(model_tree, cp = min_cp )
prp(model_tree_prune) # new new tree is unchanged
printcp(model_tree_prune)

# Regress cnt on only hr and lag variables we will get the same tree as model_tree
model_tree_simple <- rpart(data=data_training, formula = cnt ~ hr + lagged_cnt,
                           method="anova")
prp(model_tree_simple)
printcp(model_tree_simple)

# Predict the number of rental bikes
pred_tree <- predict(model_tree)
mse_tree <- mean((data$cnt - pred_tree)^2)

# Compares the true (black) and the predicted values (orange)

library(ggplot2)

ggplot(data=data_training, aes(x=lagged_cnt, y=cnt))+
  geom_point(size=0.5) +
  geom_point(aes(x=lagged_cnt, y=pred_tree), color="#f1b147") +
  labs(
    x = "Lagged count",
    y = "Count of total rental bikes") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))

# Plot the predictions against the true values
ggplot()+
  geom_point(aes(x=pred_tree, y=data_training$cnt)) +
  geom_abline(slope = 1, intercept = 0, linetype="dashed") +
  labs(
    x = "Predicted count",
    y = "True count") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))
  

#-------------------------------------------------------

# PCA with scaled dependent variable

# Get standardized data
dt_pca <- data.table(scale(data_training))

# Only features to avoid information leakage on dependent
feature_pca <- data.table(scale(data_training[, -"cnt"]))
model_feature_pca <- prcomp(feature_pca)

# The eigenvectors
model_feature_pca$rotation
# The new 22 coordinates in the PC coordinate system 
model_feature_pca$x

# New training set with principal components (cnt = dt_pca$cnt)
train_set_pca <- data.frame(cnt = dt_pca$cnt, model_feature_pca$x)

# Tree model with PCA
model_tree_PCA <- rpart(formula = cnt ~ ., data=train_set_pca, method="anova")
prp(model_tree_PCA)


# Predict the number of rental bikes (mean((dt_pca$cnt...)
pred_tree_PCA <- predict(model_tree_PCA)
mse_tree_PCA <- mean((dt_pca$cnt - pred_tree_PCA)^2)

#-----------------------------------

# PCA without scaling the dependent variable

# New training set with principal components (cnt = data_training$cnt)
train_set_pca_nonscale <- data.frame(cnt = data_training$cnt, model_feature_pca$x)

# Tree model with PCA
model_tree_PCA_nonscale <- rpart(formula = cnt ~ ., data=train_set_pca_nonscale, method="anova")
prp(model_tree_PCA_nonscale)


# Predict the number of rental bikes (mean((data_training$cnt...)
pred_tree_PCA_nonscale <- predict(model_tree_PCA_nonscale)
mse_tree_PCA_nonscale <- mean((data_training$cnt - pred_tree_PCA_nonscale)^2)
