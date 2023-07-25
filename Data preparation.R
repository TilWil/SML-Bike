# Data preparation

# source("Packages.R")

rm(list = ls())

library(dplyr)
library(data.table)

# data <- read.csv("C:\\Users\\tilma\\Documents\\Uni\\Master Economics\\Machine Learning\\Statistical Machine Learning\\Data\\hour.csv")
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

# drop "dteday" as all information already contained in other variables
data$dteday <- NULL

# DUMMYS

# Convert "season" to a factor with custom labels
data$season <- factor(data$season, levels = 1:4, labels = c("Spring", "Summer", "Fall", "Winter"))

# Convert "weekday" to a factor with custom labels and relevel the factor to start from Monday (1)
data$weekday <- factor(data$weekday, levels = 0:6, labels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
data$weekday <- relevel(data$weekday, ref = "Monday")

# Create dummy variables for "season" and "weekday"
season_dummies <- model.matrix(~ season - 1, data = data)
weekday_dummies <- model.matrix(~ weekday - 1, data = data)


# Update column names with the desired names
colnames(season_dummies) <- levels(data$season)
colnames(weekday_dummies) <- levels(data$weekday)


# Combine the dummy variables and remove the original variables
data <- cbind(data, season_dummies, weekday_dummies)
data <- data[, !"season", with = FALSE]
data <- data[, !"weekday", with = FALSE]
data <- data[, !"instant", with = FALSE]
data <- data[, !"registered", with = FALSE] # need to exclude both as they are 
data <- data[, !"casual", with = FALSE]     # sub-measures of the label "cnt"



# Convert all variables to numeric using lapply and as.numeric
data[] <- lapply(data, as.numeric)

# Create Test and Training dataset (80% of data) and test dataset (20% of data)
training_sample <- data[,sample(.N, floor(.N*.80))]
data_training <- data[training_sample]
data_test <- data[-training_sample]


  


