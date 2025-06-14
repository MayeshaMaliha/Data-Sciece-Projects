# Breast Cancer Prediction using ML models in R

# --- SETUP ---
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
library(gam)  # Needed for gamLoess

data(brca)

# --- DATA EXPLORATION ---
str(brca)
table(brca$y)
any(is.na(brca$x))
dim(brca$x)
length(brca$y)
summary(brca$x)
mean(brca$y == "M")  # Proportion of malignant cases

# --- SCALING FEATURES ---
means <- colMeans(brca$x)
sds <- colSds(brca$x)
x_centered <- sweep(brca$x, 2, means, "-")
x_scaled <- sweep(x_centered, 2, sds, "/")

# --- PCA (Optional Visualization) ---
pca <- prcomp(x_scaled)
summary(pca)

# --- DATA SPLITTING ---
set.seed(1)
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index, ]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index, ]
train_y <- brca$y[-test_index]

train_set <- data.frame(train_x)
train_set$diagnosis <- train_y
test_set <- data.frame(test_x)

# --- MODEL 1: Logistic Regression ---
model_glm <- train(diagnosis ~ ., data = train_set, method = "glm", family = "binomial")
glm_preds <- predict(model_glm, newdata = test_set)

# --- MODEL 2: Loess (gamLoess) ---
model_loess <- train(diagnosis ~ ., data = train_set, method = "gamLoess")
loess_preds <- predict(model_loess, newdata = test_set)

# --- MODEL 3: K-Nearest Neighbors ---
set.seed(7)
k_grid <- data.frame(k = seq(3, 21, 2))
model_knn <- train(diagnosis ~ ., data = train_set, method = "knn", tuneGrid = k_grid)
knn_preds <- predict(model_knn, newdata = test_set)

# --- MODEL 4: Random Forest ---
set.seed(9)
rf_grid <- data.frame(mtry = c(3, 5, 7, 9))
model_rf <- train(diagnosis ~ ., data = train_set, method = "rf", tuneGrid = rf_grid, importance = TRUE)
rf_preds <- predict(model_rf, newdata = test_set)

# --- ENSEMBLE VOTING ---
ensemble <- cbind(glm = glm_preds == "B", loess = loess_preds == "B", rf = rf_preds == "B", knn = knn_preds == "B")
ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "B", "M")
ensemble_preds <- factor(ensemble_preds, levels = levels(test_y))

# --- EVALUATION ---
acc_glm <- mean(glm_preds == test_y)
acc_loess <- mean(loess_preds == test_y)
acc_knn <- mean(knn_preds == test_y)
acc_rf <- mean(rf_preds == test_y)
acc_ensemble <- mean(ensemble_preds == test_y)

accuracy_table <- data.frame(
  Model = c("Logistic Regression", "Loess", "kNN", "Random Forest", "Ensemble"),
  Accuracy = round(c(acc_glm, acc_loess, acc_knn, acc_rf, acc_ensemble), 4)
)

print(accuracy_table)
best_model <- accuracy_table[which.max(accuracy_table$Accuracy), ]
cat("\nBest Model:", best_model$Model, "with accuracy", best_model$Accuracy, "\n")
