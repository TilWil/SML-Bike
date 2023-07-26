# Linear regression model

# source("Data preparation.R")

library(stargazer)
library(data.table)
library(ggplot2)
library(caret)

# baseline linear model
  lm_basic <- lm(cnt ~ yr + mnth + hr + holiday + workingday +
                 Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                 Friday + Saturday + Sunday + weathersit + temp + atemp + hum + windspeed,
                 data = data) 
  residuals <- lm_basic$residuals
  empirical_loss_basic <- sum(residuals^2)
  
  # Adding a lagged variabl
  lm_lagged <- lm(cnt ~ lagged_cnt + yr + mnth + hr + holiday  + workingday +
                  weathersit + temp + atemp + hum + windspeed + 
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday,
                  data = data) 
  residuals <- lm_lagged$residuals
  empirical_loss_lagged <- sum(residuals^2)
  
  # Including squared terms
  data$sq_mnth <- data$mnth^2
  data$sq_temp <- data$temp^2
  data$sq_hr <- data$hr^2
  
  lm_sq <- lm(cnt ~ lagged_cnt + yr + mnth + sq_mnth + hr + sq_hr + holiday + 
                 Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                 Friday + Saturday + Sunday + workingday + weathersit + temp + sq_temp + atemp + 
                 hum + windspeed,data = data)
  summary(lm_sq)
  residuals <- lm_sq$residuals
  empirical_loss_sq <- sum(residuals^2)
  
  # including interaction terms to the lagged model
  lm_interaction <- lm(cnt ~ lagged_cnt + yr + mnth + hr + holiday  + workingday +
                  (workingday*temp) + weathersit + temp + atemp + hum + windspeed + 
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday +
                  (mnth*temp),data = data) 
  residuals <- lm_interaction$residuals
  empirical_loss_interaction <- sum(residuals^2)
  
  # Including all adjustments
  lm_full <- lm(cnt ~ lagged_cnt + yr + mnth + sq_mnth + hr + sq_hr + holiday + workingday +
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday + weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp),
                  data = data)
  residuals <- lm_full$residuals
  empirical_loss_full <- sum(residuals^2)


  
  stargazer(lm_basic, lm_lagged, lm_interaction, lm_full, type = "text", style = "aer")
  # F-stat in all cases very high - R² doesnt improve much with complexity, except 
  # the lagged add-on
 

# Investigate on losses 

# 1. Create data.table to store the results
result <- data.table(
  " " = "empirical loss",
  loss_basic = empirical_loss_basic,
  loss_lagged = empirical_loss_lagged,
  loss_interaction = empirical_loss_interaction,
  loss_sq = empirical_loss_sq,
  loss_full = empirical_loss_full
)





# RERUN FULL MODEL WITH PCA DATA
  
  # Need to include squared terms
  data_PC18$sq_mnth <- data_PC18$mnth^2
  data_PC18$sq_temp <- data_PC18$temp^2
  data_PC18$sq_hr <- data_PC18$hr^2
  
  # Including all adjustments
  lm_pca_full <- lm(cnt ~ lagged_cnt + yr + mnth + sq_mnth + hr + sq_hr + holiday + workingday +
                  Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                  Friday + Saturday + Sunday + weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp),
                data = data_PC18)
  residuals <- lm_pca_full$residuals
  empirical_loss_pca_full <- sum(residuals^2)
  
  # compare pca to non-pca
  stargazer(lm_full, lm_pca_full, type = "text", style = "aer") # R² = 1 with pca
  
  
  # Inspect on empirical loss again
  result <- data.table(
    " " = "empirical loss",
    loss_basic = empirical_loss_basic,
    loss_lagged = empirical_loss_lagged,
    loss_interaction = empirical_loss_interaction,
    loss_sq = empirical_loss_sq,
    loss_full = empirical_loss_full,
    loss_pca_full = empirical_loss_pca_full
    
  )
  
  result
  
# CROSS-VALIDATION
  
  # Function that splits "data" into K Folds 
  
  split_K <- function(data, folds){
    data$id <- ceiling(sample(1:nrow(data), replace = FALSE, nrow(data)) / # Create an id variable
                         (nrow(data) / folds))         # which randomly assigns each row to a fold 
    return(data)
  }
  
  # Randomly assign data points to K folds
  dt_training <- split_K(data_training, K)
  dt_training$sq_mnth <- dt_training$mnth^2
  dt_training$sq_temp <- dt_training$temp^2
  dt_training$sq_hr <- dt_training$hr^2
  
  # Train object for the caret package specifies method and number of runs
  
  
  train_control <- trainControl(method = "cv", number = 5)
  
  lm_caret <- train(
    cnt  ~ lagged_cnt + yr + mnth + sq_mnth + hr + sq_hr + holiday + workingday +
      Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
      Friday + Saturday + Sunday + weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp),
    data = dt_training[,-"id"],
    method = "lm",
    trControl = train_control)
  summary(lm_caret)
  
  lm_caret$results 
    
# END








# Non-required additional steps:


### EXPECTED LOSS by sampling the data

    # Draw 10 samples out of the data and perform the full linear model on each
      S <- 10 # number of samples
      N <- 300 # sample size
      
      samples <- list()
      models <- data.frame()
      losses <- data.frame()
      
      # Draw S samples
      for (i in 1:S) {
        samples[[i]] <- data[sample(nrow(data), N), ]
      }
      
      # Run a model on each sample
      for (i in 1:S) {
        lm_losses <- lm(formula = cnt ~  yr + mnth + sq_mnth + hr + holiday  + workingday +
                        weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp) +
                        Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                        Friday + Saturday + Sunday, data = samples[[i]])
        # store model parameters
        models <- rbind(models, lm_losses$coefficients)
        # store model performance on sample and population
        losses <- rbind(losses, data.table("model" = i, 
                                           "empirical_loss" = sum(mean(lm_losses$residuals^2)),
                                           "expected_loss" = sum(mean((data$cnt - predict(lm_losses, newdata = data))^2))))
      }
    
      
      losses <- melt(losses, id.vars = "model", variable.name = "Loss", value.name = "MSE")
      
      ggplot(data = losses, aes(y=MSE, x=model, fill= Loss)) +
        geom_bar(position="dodge", stat="identity") +
        geom_hline(yintercept = sum(mean(lm(formula = cnt ~  yr + mnth + sq_mnth + hr + holiday  + workingday +
                        weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp) +
                        Spring + Summer + Fall + Winter + Monday + Tuesday + Wednesday + Thursday +
                        Friday + Saturday + Sunday,data=data)$residuals^2)),
                   color="#4CA7DE", linetype="dashed", size=1) +
        labs(
          x = "Models",
          y = "MSE") +
        scale_fill_manual(values=c("lightgrey","darkgrey")) +
        scale_x_continuous(breaks=seq(1, S, by = 1)) +
        theme_minimal() +
        theme(axis.line = element_line(color = "#000000"),
              axis.text.y=element_blank())
  
    # Expected loss here refers to the "unseen" non-sampled data points - empirical loss
    # to the mean-squared-error on the sampled data point
  