# Packages
install.packages("dplyr")
install.packages("data.table")
install.packages("ggplot2")
install.packages("factoextra")
install.packages("NbClust")
install.packages("e1071")
install.packages("CRAN")
install.packages("kernlab")
install.packages("keras")
install.packages("viridisLite")
install.packages("plot3D")
install.packages("tensorflow")
install.packages("reticulate")
install.packages("miniconda")
install.packages("tidymodels")
install.packages("ada")
install.packages("caret")
install.packages("neuralet")



# To install tensorflow do either:
library(reticulate)
path_to_python <- install_python()
virtualenv_create("r-reticulate", python = path_to_python)
library(tensorflow)
install_tensorflow(envname = "r-reticulate")
# Or:
install.packages("keras")
library(keras)
install_keras(envname = "r-reticulate")
library(tensorflow)
tf$constant("Hello Tensorflow!")