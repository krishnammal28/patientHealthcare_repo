# Q7. Predictive analysis in R: diagnose whether a tumour is malignant (M) or benign (B)
# Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), 569 patients, 30 features.
# Run: Rscript 07_breast_cancer_prediction.R
# Packages: install.packages(c("caret", "e1071", "randomForest"))

library(caret)
library(randomForest)

set.seed(123)

# 1. Load data directly from the UCI repository (internet needed).
#    If offline, download wdbc.data once and set `path` to the local file.
path <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cols <- c("id", "diagnosis",
          paste0(rep(c("radius", "texture", "perimeter", "area", "smoothness",
                       "compactness", "concavity", "concave_points", "symmetry",
                       "fractal_dim"), 3),
                 rep(c("_mean", "_se", "_worst"), each = 10)))
bc <- read.csv(path, header = FALSE, col.names = cols)

# 2. Preprocessing
bc$id <- NULL
bc$diagnosis <- factor(bc$diagnosis, levels = c("B", "M"),
                       labels = c("Benign", "Malignant"))
cat("Missing values:", sum(is.na(bc)), "\n")
print(table(bc$diagnosis))

# 3. Train / test split (70 / 30)
idx   <- createDataPartition(bc$diagnosis, p = 0.7, list = FALSE)
train <- bc[idx, ]
test  <- bc[-idx, ]

# 4. Logistic regression (on scaled features)
pre   <- preProcess(train[, -1], method = c("center", "scale"))
trs   <- predict(pre, train)
tes   <- predict(pre, test)
glm_model <- suppressWarnings(glm(diagnosis ~ ., data = trs, family = binomial))
glm_prob  <- suppressWarnings(predict(glm_model, tes, type = "response"))
glm_pred  <- factor(ifelse(glm_prob > 0.5, "Malignant", "Benign"),
                    levels = levels(bc$diagnosis))
cat("\n--- Logistic Regression ---\n")
print(confusionMatrix(glm_pred, test$diagnosis, positive = "Malignant"))

# 5. Random forest
rf_model <- randomForest(diagnosis ~ ., data = train, ntree = 300, importance = TRUE)
rf_pred  <- predict(rf_model, test)
cat("\n--- Random Forest ---\n")
print(confusionMatrix(rf_pred, test$diagnosis, positive = "Malignant"))

# 6. Most important predictors
cat("\nTop 10 important features:\n")
imp <- importance(rf_model)[, "MeanDecreaseGini"]
print(head(sort(imp, decreasing = TRUE), 10))

# 7. Predict for a new patient (first test row as an example)
new_patient <- test[1, -1]
cat("\nDiagnosis predicted for a new patient:",
    as.character(predict(rf_model, new_patient)), "\n")
