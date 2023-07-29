library(rpart)
library(rpart.plot)

# Tree regression using all dependent variable except for lagged count
model_tree_nonlag <- rpart(data=data, formula = cnt ~ yr + mnth + hr +
                      holiday + workingday +
                      Spring + Summer + Fall + Winter +
                      Monday + Tuesday + Wednesday + Thursday +
                      Friday + Saturday + Sunday +
                      weathersit + temp + atemp + hum + windspeed,
                    method="anova")
prp(model_tree_nonlag)
printcp(model_tree_nonlag)

# Add lagged count, exclude workingday, Winter, Sunday
model_tree_full <- rpart(data=data, formula = cnt ~ yr + mnth + hr +
                      holiday + lagged_cnt +
                      Spring + Summer + Fall +
                      Monday + Tuesday + Wednesday + Thursday +
                      Friday + Saturday +
                      weathersit + temp + atemp + hum + windspeed,
                    method="anova")
prp(model_tree_full)

# Predict the number of rental bikes
pred_tree_full <- predict(model_tree_full, data[, c("yr", "mnth", "hr", "holiday", "lagged_cnt", "Spring", "Summer", "Fall", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "weathersit", "temp", "atemp", "hum", "windspeed")])
mse_tree_full <- mean((data$cnt - pred_tree_full)^2)

# Tabular and a graphical overview of the cp parameter, which captures model complexity
printcp(model_tree_full)
plotcp(model_tree_full)

# Choosing the cp value that minimizes the cross-validation error (xerror): the last row in the cp table
min_cp <- model_tree$cptable[which.min(model_tree_full$cptable[, "xerror"]), "CP"]

model_tree_prune <- prune(model_tree_full, cp = min_cp )
prp(model_tree_prune)
printcp(model_tree_prune)

# Regress cnt on only hr and lag variables we will get the same tree as model_tree_full
model_tree_simple <- rpart(data=data, formula = cnt ~ hr + lagged_cnt,
                    method="anova")
prp(model_tree_simple)
printcp(model_tree_simple)

# Predict cnt
pred_tree_simple <- predict(model_tree_simple, data[, c("hr","lagged_cnt")])
mse_tree_simple <- mean((data$cnt - pred_tree_simple)^2)

# Linear model
lm_simple <- lm(cnt ~ hr + lagged_cnt, data = data)
mse_lm_simple <- mean(lm_simple$residuals^2)
#pred_lm_simple <- predict(lm_simple, data[, c("hr","lagged_cnt")])
#MSE_lm <- mean((data$cnt - pred_lm_simple)^2)

lm_full <- lm(cnt ~ yr + mnth + hr +
                holiday + lagged_cnt +
                Spring + Summer + Fall +
                Monday + Tuesday + Wednesday + Thursday +
                Friday + Saturday +
                weathersit + temp + atemp + hum + windspeed,
              data = data)
summary(lm_full)
mse_lm_full <- mean(lm_full$residuals^2)

# Create a data frame with the model names and their corresponding MSE values
result_table <- data.frame(
  Model = c("model_tree_simple", "model_tree_full", "lm_simple", "lm_full"),
  MSE = c(mse_tree_simple, mse_tree_full, mse_lm_simple, mse_lm_full)
)

print(result_table)

# Compares the true (black) and the predicted values (orange)

library(ggplot2)

ggplot(data=data, aes(x=lagged_cnt, y=cnt))+
  geom_point(size=0.5) +
  geom_point(aes(x=lagged_cnt, y=pred_tree_simple), color="#f1b147") +
  labs(
    x = "Lagged count",
    y = "Count of total rental bikes") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))

# Plot the predictions against the true values
ggplot()+
  geom_point(aes(x=pred_tree_simple, y=data$cnt)) +
  geom_abline(slope = 1, intercept = 0, linetype="dashed") +
  labs(
    x = "Predicted count",
    y = "True count") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))
  

#-------------------------------------------------------

# ENSEMBLE

# Bagging

library(randomForest)

model_bag <- randomForest(formula = cnt ~ yr + mnth + hr + holiday +
                            lagged_cnt + Spring + Summer + Fall +
                            Monday + Tuesday + Wednesday + Thursday +
                            Friday + Saturday + weathersit + temp +
                            atemp + hum + windspeed,
                          data=data_training, 
                          mtry=(ncol(data_training)-4), ntree=500)

model_bag_full <- randomForest(formula = cnt ~ .,
                          data=data_training, 
                          mtry=(ncol(data_training)-1), ntree=500)
print(model_bag_full)
# Random Forest

#  Restricts the number of random variables (columns) considered on every split to 22/3=7

model_rf <- randomForest(cnt ~ ., data=data_training, ntree=500)
print(model_rf)

# set the number of random inputs selected at each split
tunerf <- tuneRF(x=data_training[,-"cnt"], y=data_training[,cnt], ntreeTry = 500)

# Gradient Boosting

library(gbm)

model_gbm <- gbm(cnt ~ ., 
                 data=data_training, 
                 distribution="gaussian",
                 interaction.depth=1,
                 n.trees=500, 
                 shrinkage=0.1,
                 cv.folds = 5)
print(model_gbm)

# Estimates the optimal number of boosting iterations and plot the performance measures
gbm.perf(model_gbm, method = "cv")
# Training error in black, validation error in green, the blue dashed line shows what is the optimal number of iterations according to the metric and validation procedure used