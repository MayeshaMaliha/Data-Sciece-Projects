# Breast Cancer Prediction using Machine Learning (R + caret)

This project uses the `brca` dataset from the `dslabs` package to predict whether breast cancer biopsy samples are **malignant (M)** or **benign (B)** using various machine learning models.

## 📊 Dataset
- Source: `dslabs::brca`
- `x`: Matrix of numeric features (e.g., radius, texture, perimeter, area, etc.)
- `y`: Factor variable — "B" (benign) or "M" (malignant)

## 🔍 Goal
Train, evaluate, and compare the performance of multiple classification models, and build an ensemble for improved predictive accuracy.

## 💻 Models Used
- Logistic Regression (GLM)
- Loess (gamLoess)
- k-Nearest Neighbors (kNN)
- Random Forest (RF)
- Majority-Vote Ensemble (combining the four above)

## 🧪 Workflow
1. Load and explore the data
2. Scale the features (zero mean, unit variance)
3. Split into training and test sets (80/20)
4. Train models using `caret::train()`
5. Evaluate each model's accuracy on the test set
6. Combine predictions into an ensemble using majority voting
7. Compare all models in a summary table

## 📈 Accuracy Output Example
| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 0.9569   |
| Loess              | 0.9569   |
| kNN                | 0.9522   |
| Random Forest      | 0.9655   |
| **Ensemble**       | **0.9655**   |

## 📦 Requirements
Install the following R packages:
```r
install.packages(c("dslabs", "caret", "tidyverse", "matrixStats", "gam"))
