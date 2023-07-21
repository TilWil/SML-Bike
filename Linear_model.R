# Linear regression model
library(stargazer)
library(data.table)
library(ggplot2)

# baseline linear model
  lm_basic <- lm(cnt ~ season + yr + mnth + hr + holiday + weekday + workingday +
                 weathersit + temp + atemp + hum + windspeed + mnth,data = data) 
  residuals <- lm_basic$residuals
  empirical_loss_basic <- sum(residuals^2)
  
  # Adding a lagged variabl
  lm_lagged <- lm(cnt ~ lagged_cnt + season + yr + mnth + hr + holiday + weekday + workingday +
                   weathersit + temp + atemp + hum + windspeed + mnth,data = data) 
  residuals <- lm_lagged$residuals
  empirical_loss_lagged <- sum(residuals^2)
  
  # Including squared terms
  data$sq_mnth <- data$mnth^2
  data$sq_temp <- data$temp^2
  data$sq_hr <- data$hr^2
  
  lm_sq <- lm(cnt ~ lagged_cnt + season + yr + mnth + sq_mnth + hr + sq_hr + holiday + 
                 weekday + workingday + weathersit + temp + sq_temp + atemp + 
                 hum + windspeed,data = data)
  summary(lm_sq)
  residuals <- lm_sq$residuals
  empirical_loss_sq <- sum(residuals^2)
  
  # including interaction terms to the lagged model
  lm_interaction <- lm(cnt ~ lagged_cnt + season + yr + mnth + hr + holiday + weekday + workingday +
                  (workingday*temp) + weathersit + temp + atemp + hum + windspeed + 
                  (mnth*temp),data = data) 
  residuals <- lm_interaction$residuals
  empirical_loss_interaction <- sum(residuals^2)
  
  # Including all adjustments
  lm_full <- lm(cnt ~ lagged_cnt + season + yr + mnth + sq_mnth + hr + sq_hr + holiday + weekday + workingday +
                  weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp),data = data)
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



### Expected loss by sampling the data - (later will bootstrap)

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
    lm_losses <- lm(formula = cnt ~ season + yr + mnth + sq_mnth + hr + holiday + weekday + workingday +
                      weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp), data = samples[[i]])
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
    geom_hline(yintercept = sum(mean(lm(formula = cnt ~ season + yr + mnth + sq_mnth + hr + holiday + weekday + workingday +
                                             weathersit + temp + sq_temp + atemp + hum + windspeed + (mnth*temp),data=data)$residuals^2)),
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


















# Same on no catergorigal variables data
lm_basic_sub <- lm(cnt ~ yr + mnth + hr + temp + atemp + hum + windspeed + mnth,data = dt_nocat) 
summary(lm_basic_sub)

sq_mnth <- data$mnth^2
sq_temp <- data$temp^2

lm_sq_sub <- lm(cnt ~ yr + mnth + hr + temp + atemp + hum + windspeed + mnth + 
                  sq_mnth + sq_temp,data = dt_nocat)
summary(lm_sq_sub)


lm_interaction_sub <- lm(cnt ~ yr + mnth + hr + temp + atemp + hum + windspeed + mnth +
                       (mnth*temp),data = dt_nocat) 
summary(lm_interaction_sub)

lm_full_sub <- lm(cnt ~ yr + mnth + hr + temp + atemp + hum + windspeed + mnth +
               sq_temp + sq_mnth + (mnth*temp), data = dt_nocat)
summary(lm_full_sub)
stargazer(lm_basic_sub, lm_sq_sub, lm_interaction_sub, lm_full_sub, type = "text", style = "aer")
# no improvements in terms of goodness-of-fit