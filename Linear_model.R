# Linear regression model
# Authors: Hong Le, Tilman von Samson
# 31.07.2023

source("Data preparation.R")

# Content
# 1. Linear Models with varying complexity
# 2. Rerun Full-Model with PCA Data
# 3. Empirical Loss of all models 
# 4. Cross-Validation
# 5. Expected Loss

library(stargazer)
library(data.table)
library(ggplot2)
library(caret)

# 1. Linear Model set-ups with varying complexity

# 1.a) Basic-Linear-Model
  lm_basic <- lm(cnt ~ yr + mnth + hr + holiday  +
                 Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                 Friday + Saturday + Sunday + weathersit + temp + atemp + hum + windspeed,
                 data = data_training)  # Normalized data for comparability to PCA reduced model
  residuals <- lm_basic$residuals
  empirical_loss_basic <- sum(residuals^2)
  
  # 1.b) Lagged-Dependent-Variable-Model
  lm_lagged <- lm(cnt ~ lagged_cnt + yr + mnth + hr + holiday   +
                  weathersit + temp + atemp + hum + windspeed + 
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday,
                  data = data_training) 
  residuals <- lm_lagged$residuals
  empirical_loss_lagged <- sum(residuals^2)
  
  # 1. c) Squared-Terms-Model
  lm_sq <- lm(cnt ~ lagged_cnt + yr + mnth + I(mnth^2) + hr + I(hr^2) + holiday + 
                 Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                 Friday + Saturday + Sunday  + weathersit + temp + I(temp^2) + atemp + 
                 hum + windspeed,data = data_training)
  summary(lm_sq)
  residuals <- lm_sq$residuals
  empirical_loss_sq <- sum(residuals^2)
  
  # 1.d) Interaction-Terms-Model
  lm_interaction <- lm(cnt ~ lagged_cnt + yr + mnth + hr + holiday + (holiday*weathersit)+
                  weathersit + temp + atemp + hum + windspeed + 
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday +
                  (mnth*temp),data = data_training) 
  residuals <- lm_interaction$residuals
  empirical_loss_interaction <- sum(residuals^2)
  
  # 1.e) Full-Model
  lm_full <- lm(cnt ~ lagged_cnt + yr + mnth + I(mnth^2) + hr + I(hr^2) + holiday  + (holiday*weathersit)+
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday + weathersit + temp + I(temp^2) + atemp + hum + windspeed + (mnth*temp),
                  data = data_training)
  residuals <- lm_full$residuals
  empirical_loss_full <- sum(residuals^2)
  
  # Train same model on normalized data for comparision with PCA Model
  lm_full_norm <- lm(cnt ~ lagged_cnt + yr + mnth + I(mnth^2) + hr + I(hr^2) + holiday  +
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday + weathersit + temp + I(temp^2) + atemp + hum + windspeed + (mnth*temp),
                data = data_training_norm)

  # 1.f) Print results
  stargazer(lm_basic, lm_lagged, lm_interaction, lm_full, type = "text", style = "aer") # Can only include 4 models
  # F-stat in all cases very high - R² doesnt improve much with complexity,  
  # except the lagged add-on
 


# 2. RERUN FULL MODEL WITH PCA DATA
  
  # Including all adjustments
  lm_pca_full <- lm(data_training$cnt ~ lagged_cnt + yr + mnth + I(mnth^2) + hr + I(hr^2) + holiday  +
                  Spring + Summer + Fall + Winter + workingday + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday + weathersit + temp + I(temp^2) + atemp + hum + windspeed + (mnth*temp),
                data = feature_PC17)
  residuals <- lm_pca_full$residuals
  empirical_loss_pca_full <- sum(residuals^2)
  
  # compare pca to non-pca
  stargazer(lm_full, lm_pca_full, type = "text", style = "aer") # R² = 1 with pca
  
  

# 3. Inspect on EMPIRICAL loss of all models
  Empirical_loss <- data.table(
    " " = "empirical loss",
    loss_basic = empirical_loss_basic,
    loss_lagged = empirical_loss_lagged,
    loss_interaction = empirical_loss_interaction,
    loss_sq = empirical_loss_sq,
    loss_full = empirical_loss_full,
    loss_pca_full = empirical_loss_pca_full)
  
  show(Empirical_loss)
  
# 4. CROSS-VALIDATION
  # (does not make much sense for the linear model, as overfitting no big issue here)

  # Train object for the caret package specifies method and number of runs
  
  train_control <- trainControl(method = "cv", number = 5)
  
  lm_cv <- train(
    cnt  ~ lagged_cnt + yr + mnth + I(mnth^2) + hr + I(hr^2) + holiday  +
      Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
      Friday + Saturday + Sunday + weathersit + temp + I(temp^2) + atemp + hum + windspeed + (mnth*temp),
    data = data_training,
    method = "lm",
    trControl = train_control)
  summary(lm_cv)
  
  
# 5. EXPECTED LOSS

  Expected_Loss_lm_cv <- mean((data_test[, cnt]-
                                  predict(lm_cv, newdata = data_test))^2)
  Expected_Loss_lm_full <- mean((data_test[, cnt]-
                                    predict(lm_full, newdata = data_test))^2)
  Expected_Loss_lm_pca_full <- mean((data_test[, cnt]-
                                   predict(lm_pca_full, newdata = data_test))^2)
  
  Expected_Loss_lm_cv_norm <- mean((data_test_norm[, cnt]-
                                 predict(lm_cv, newdata = data_test_norm))^2)
  Expected_Loss_lm_full_norm <- mean((data_test_norm[, cnt]-
                                   predict(lm_full_norm, newdata = data_test_norm))^2)
  Expected_Loss_lm_pca_full_norm <- mean((data_test_norm[, cnt]-
                                       predict(lm_pca_full, newdata = data_test_norm))^2)
  
  Expected_loss <- data.table(
    " " = "expected loss",
    Loss_lm_cv = Expected_Loss_lm_cv,
    Loss_lm_full = Expected_Loss_lm_full,
    Loss_lm_pca_full = Expected_Loss_lm_pca_full,
    Loss_dnn = Expected_loss_dnn,
    Loss_dnn_cv = Expected_loss_dnn_cv,
    Loss_dnn_cv2 = Expected_loss_dnn_cv2,
    Loss_dnn_pca = Expected_loss_dnn_pca,
    Loss_dnn_pca_on_pca = Expected_loss_dnn_pca_on_pca)
  
  show(Expected_loss)

  # END

