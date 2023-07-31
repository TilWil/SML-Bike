# EXPECTED LOSS SUMMARY
# Authors: Hong Le, Tilman von Samson
# 31.07.2023

# source("Linear_model.R")
# source("Neural_Network.R")

# All Expected loss calculations from the slides collected

Expected_Loss_lm_cv <- mean((data_test[, cnt]-
                               predict(lm_cv, newdata = data_test))^2)
Expected_Loss_lm_full <- mean((data_test[, cnt]-
                                 predict(lm_full, newdata = data_test))^2)
Expected_Loss_lm_pca_full <- mean((data_test[, cnt]-
                                     predict(lm_pca_full, newdata = data_test))^2)
# With normalized test data
Expected_Loss_lm_cv_norm <- mean((data_test_norm[, cnt]-
                                    predict(lm_cv, newdata = data_test_norm))^2)
Expected_Loss_lm_full_norm <- mean((data_test_norm[, cnt]-
                                      predict(lm_full, newdata = data_test_norm))^2)
Expected_Loss_lm_pca_full_norm <- mean((data_test_norm[, cnt]-
                                          predict(lm_pca_full, newdata = data_test_norm))^2)

# Model dnn
dnn_test_predictions <- predict(dnn_model, as.matrix(test_features))
Expected_loss_dnn <- mean((as.matrix(test_labels) - dnn_test_predictions)^2)

# Model dnn_cv
dnn_cv_test_predictions <- predict(dnn_cv_model, as.matrix(test_features))
Expected_loss_dnn_cv <- mean((as.matrix(test_labels) - dnn_cv_test_predictions)^2)

# Model dnn_cv_pca
dnn_cv_pca_test_predictions <- predict(dnn_cv_pca_model, as.matrix(test_features))
Expected_loss_dnn_cv_pca <- mean((as.matrix(test_labels) - dnn_cv_pca_test_predictions)^2)

# Model dnn_pca 
dnn_pca_test_predictions <- predict(dnn_pca_model, as.matrix(test_features))
Expected_loss_dnn_pca <- mean((as.matrix(test_labels) - dnn_pca_test_predictions)^2)

# CV Model 2
Expected_loss_dnn_cv2 <- mean(as.matrix((test_labels_norm - predict(dnn_cv_model2, test_features_norm))^2))

# Boosted tree
Expected_Loss_boost_cv <- mean((data_test[, cnt]-
                                     predict(model_gbm, newdata = data_test))^2)

# Create summary Table

Expected_loss <- data.table(
  " " = "expected loss",
  Loss_lm_cv = Expected_Loss_lm_cv,
  Loss_lm_full = Expected_Loss_lm_full,
  Loss_lm_pca_full = Expected_Loss_lm_pca_full,
  Loss_lm_cv_norm = Expected_Loss_lm_cv_norm,
  Loss_lm_full_norm = Expected_Loss_lm_full_norm,
  Loss_lm_pca_full_norm = Expected_Loss_lm_pca_full_norm,
  Loss_dnn = Expected_loss_dnn,
  Loss_dnn_cv = Expected_loss_dnn_cv,
  Loss_dnn_cv2 = Expected_loss_dnn_cv2,
  Loss_dnn_pca = Expected_loss_dnn_pca,
  Loss_boosting_cv = Expected_Loss_boost_cv)

show(Expected_loss)
# Save results
write.csv(Expected_loss, "C:\\Users\\tilma\\Documents\\Uni\\Master Economics\\Machine Learning\\Statistical Machine Learning\\Expected_loss.csv")

### END

