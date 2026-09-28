# =================================================================
# Comparative Evaluation of Machine Learning Algorithms for
# Predicting Live Body Weight from Morphometric Traits in Cattle
#
# Consolidated, reproducible R script
# Source data: Bokolo_ML.xlsx (177 raw cattle records)
# Modeling data: cattle_final_168.xlsx (168 cattle, 21 morphometric
#                traits + sex, target = Estimated body weight)
# =================================================================


# =================================================================
# STEP 0: DATA CLEANING (performed once, outside R, documented here
# for full reproducibility -- this produced cattle_final_168.xlsx)
# =================================================================
# Starting from the raw file (Bokolo_ML.xlsx, 177 animals, 29 columns):
#   1. Fixed a data-entry typo: row 80, "Ear width" = 140 -> corrected to 14.0
#      (every other animal fell between 9-20; 140 was almost certainly a
#      misplaced decimal/extra digit).
#   2. Dropped "Breed" (constant -- all animals were "Bokolo", zero variance,
#      not usable as a predictor).
#   3. Dropped 5 columns with heavy, STRUCTURAL missingness tied to sex/breed
#      variation rather than random measurement failure:
#        - scrotal circumference (67 missing -- males only)
#        - Udder diameter (113 missing -- females only)
#        - Horn length (87 missing), Horn tip separation (108 missing),
#          Horn base separation (115 missing) -- likely polled/hornless animals
#      Rationale: complete-casing across ALL original columns left 0 usable
#      rows, since no single animal can have both udder diameter AND scrotal
#      circumference. Dropping these 5 sex/breed-specific columns instead
#      preserved the dataset almost entirely.
#   4. Kept "sxe" (sex, M/F) as a predictor -- complete, biologically relevant.
#   5. Dropped 9 remaining rows with leftover missing values in the 22 kept
#      columns (Excel rows 10,11,12,13,26,38,80,116,178) -> 177 - 9 = 168 rows.
# Result: cattle_final_168.xlsx, 168 cattle (107 M, 61 F), 21 morphometric
# traits + sex + target, zero missing values.


# =================================================================
# STEP 1: IMPORT & SETUP
# =================================================================

setwd("C:/Users/DELL/OneDrive/Documents/ML")   # <-- EDIT THIS PATH if reused elsewhere

library(readxl)
library(caret)

cattle <- read_excel("cattle_final_168.xlsx")

str(cattle)
dim(cattle)              # 168 rows, 23 columns
sum(is.na(cattle))       # should be 0

# sxe (sex) as a factor
cattle$sxe <- as.factor(cattle$sxe)
levels(cattle$sxe)       # "F" "M"

# Clean column names (spaces -> dots) and rename target to BW
names(cattle) <- make.names(names(cattle))
names(cattle)[names(cattle) == "Estimated.body.weight"] <- "BW"
names(cattle)
# NOTE: "Distance btw eyes & nose" becomes "Distance.btw.eyes...nose" (3 dots) --
# copy-paste this exact name from names(cattle) if referencing it; don't retype
# from memory.

summary(cattle$BW)
hist(cattle$BW, main = "Distribution of Body Weight", xlab = "Body Weight (kg)",
     col = "lightblue")
# Observation: distribution is bimodal/multimodal (~100-250kg cluster, smaller
# ~350-550kg cluster) -- largely explained by sex (see boxplot below).


# =================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =================================================================

summary(cattle)

# Body weight by sex
boxplot(BW ~ sxe, data = cattle, main = "Body Weight by Sex",
        xlab = "Sex", ylab = "Body Weight (kg)", col = c("pink", "lightblue"))
# Males: higher median (~270kg) than Females (~190kg). Confirms sxe as an
# important predictor and explains the bimodal BW distribution above.
# Note: 2 high outliers (>500kg) in Females -- plausible large/old cows, kept
# in the dataset (not removed) but worth flagging in Discussion/limitations.

# Correlation of each numeric trait with BW
cor_BW <- sapply(cattle[sapply(cattle, is.numeric)], function(x) cor(x, cattle$BW))
round(sort(cor_BW, decreasing = TRUE), 3)
# Top correlates: Heart girth (0.806), Hip height (0.785), Hind leg length (0.784),
# Head length (0.743), Body length (0.720). Weakest: Rump length (0.079, ~uncorrelated).

# Full correlation matrix among predictors (pairwise multicollinearity check)
numeric_vars <- cattle[sapply(cattle, is.numeric)]
cor_matrix_cattle <- cor(numeric_vars)

high_cor <- which(abs(cor_matrix_cattle) > 0.85 & abs(cor_matrix_cattle) < 1, arr.ind = TRUE)
data.frame(
  Var1 = rownames(cor_matrix_cattle)[high_cor[, 1]],
  Var2 = colnames(cor_matrix_cattle)[high_cor[, 2]],
  Correlation = round(cor_matrix_cattle[high_cor], 2)
)
# Result: only Heart girth <-> Hip height exceeds 0.85 (r = 0.87) -- biologically
# sensible (larger-framed animals have both greater girth and hip height).

# VIF check (multivariate multicollinearity -- catches what pairwise corr misses)
library(car)
vif_model <- lm(BW ~ ., data = cattle)
vif(vif_model)
# Result: Rump heigth (15.5), Rump length (12.3), Hip height (10.6) exceed the
# VIF > 10 threshold -- multivariate multicollinearity among "rump cluster"
# traits not visible in the pairwise matrix alone.
# DECISION: keep all 21 traits + sxe in EVERY model (MLR included), for fair
# cross-algorithm comparison. Multicollinearity affects MLR coefficient
# stability/interpretation, not predictive accuracy on a held-out test set.
# Reported as a Discussion-section limitation rather than grounds for dropping
# variables.


# =================================================================
# STEP 3: TRAIN / TEST SPLIT
# =================================================================

set.seed(123)
trainIndex_cattle <- createDataPartition(cattle$sxe, p = 0.8, list = FALSE)
train_cattle <- cattle[trainIndex_cattle, ]
test_cattle  <- cattle[-trainIndex_cattle, ]

nrow(train_cattle)        # 135
nrow(test_cattle)         # 33
table(train_cattle$sxe)   # 49 F, 86 M
table(test_cattle$sxe)    # 12 F, 21 M
# Sex ratio preserved almost exactly between train and test (~36% F / 64% M).

# Numeric-encoded versions for XGBoost (the xgboost engine cannot accept
# factor columns directly, unlike lm/rf/svmRadial/nnet via caret)
train_cattle_xgb <- train_cattle
train_cattle_xgb$sxe <- ifelse(train_cattle_xgb$sxe == "M", 1, 0)

test_cattle_xgb <- test_cattle
test_cattle_xgb$sxe <- ifelse(test_cattle_xgb$sxe == "M", 1, 0)


# =================================================================
# STEP 4: MODEL TRAINING (5 ALGORITHMS, 10-FOLD CV)
# =================================================================

ctrl_cattle <- trainControl(method = "cv", number = 10)

# ----------------------------------------------------------------
# Model 1: MLR
# ----------------------------------------------------------------
set.seed(123)
model_lm_cattle <- train(BW ~ ., data = train_cattle, method = "lm",
                          trControl = ctrl_cattle)
model_lm_cattle                          # CV training: RMSE 65.13, R-sq 0.690, MAE 47.68
pred_lm_cattle <- predict(model_lm_cattle, newdata = test_cattle)
result_lm <- postResample(pred_lm_cattle, test_cattle$BW)
result_lm                                 # Test set: RMSE 54.01, R-sq 0.724, MAE 40.32

# ----------------------------------------------------------------
# Model 2: Random Forest
# ----------------------------------------------------------------
set.seed(123)
model_rf_cattle <- train(BW ~ ., data = train_cattle, method = "rf",
                          trControl = ctrl_cattle)
model_rf_cattle                          # CV training best (mtry=12): RMSE 49.62, R-sq 0.801, MAE 33.90
pred_rf_cattle <- predict(model_rf_cattle, newdata = test_cattle)
result_rf <- postResample(pred_rf_cattle, test_cattle$BW)
result_rf                                 # Test set: RMSE 52.94, R-sq 0.734, MAE 32.54

# ----------------------------------------------------------------
# Model 3: SVM (radial kernel)
# ----------------------------------------------------------------
set.seed(123)
model_svm_cattle <- train(BW ~ ., data = train_cattle, method = "svmRadial",
                           trControl = ctrl_cattle)
model_svm_cattle                         # CV training best (C=1): RMSE 59.41, R-sq 0.741, MAE 41.03
pred_svm_cattle <- predict(model_svm_cattle, newdata = test_cattle)
result_svm <- postResample(pred_svm_cattle, test_cattle$BW)
result_svm                                # Test set: RMSE 50.57, R-sq 0.768, MAE 32.69 <- BEST overall (R-sq, RMSE)

# ----------------------------------------------------------------
# Model 4: XGBoost
# ----------------------------------------------------------------
# *** IMPORTANT PACKAGE COMPATIBILITY NOTE ***
# This model requires xgboost v1.7.8.1 specifically (NOT the current CRAN
# default, v3.x). caret v7.0.1's "xgbTree" wrapper cannot read the new
# internal object format ("Altrep pointer" class) used by xgboost 2.x/3.x,
# producing all-NA CV results and:
#   "Error in modelFit$xNames <- colnames(x) :
#    ALTLIST classes must provide a Set_elt method"
# This is a confirmed, currently-open compatibility bug between the two
# packages (caret GitHub issue #1412), not a data or code problem.
#
# One-time fix applied in this project (Windows):
#   1. install.packages("pkgbuild"); pkgbuild::find_rtools()  # check compiler
#      -> if FALSE, install Rtools45 from:
#         https://cran.r-project.org/bin/windows/Rtools/rtools45/rtools.html
#      (Rtools is required to compile any R package from source on Windows;
#      this is a one-time, permanent fix for future projects too.)
#   2. remove.packages("xgboost")
#   3. Download xgboost_1.7.8.1.tar.gz manually (CRAN's archive download via
#      R sometimes times out):
#      https://cran.r-project.org/src/contrib/Archive/xgboost/xgboost_1.7.8.1.tar.gz
#   4. install.packages("<local_path>/xgboost_1.7.8.1.tar.gz", repos = NULL,
#      type = "source")
#   5. packageVersion("xgboost")  # confirm it now reports 1.7.8.1
#
library(xgboost)
# packageVersion("xgboost") should read '1.7.8.1' -- if it instead reads a
# 2.x/3.x version, the steps above need to be repeated before proceeding.

set.seed(123)
model_xgb_cattle <- train(BW ~ ., data = train_cattle_xgb, method = "xgbTree",
                           trControl = ctrl_cattle)
model_xgb_cattle
model_xgb_cattle$results[which.min(model_xgb_cattle$results$RMSE), ]
# CV training best: nrounds=100, max_depth=2, eta=0.3, gamma=0,
# colsample_bytree=0.8, min_child_weight=1, subsample=0.75
# RMSE 49.68, R-sq 0.790, MAE 34.14 -- best/near-best CV score of all 5 models

pred_xgb_cattle <- predict(model_xgb_cattle, newdata = test_cattle_xgb)
result_xgb <- postResample(pred_xgb_cattle, test_cattle_xgb$BW)
result_xgb
# Test set: RMSE 62.03, R-sq 0.645, MAE 40.02 -- WORST of all 5 models on test
# set despite best CV training fit. Classic overfitting signature: many
# tuning parameters (7) relative to small training sample (n=135) allowed
# XGBoost to fit training-specific noise that did not generalize. Key
# methodological lesson for Discussion: rank models by TEST set performance,
# not CV training performance alone.

# ----------------------------------------------------------------
# Model 5: ANN (single hidden layer, via nnet)
# ----------------------------------------------------------------
# First attempt WITHOUT predictor scaling (documented for transparency,
# not the final model used):
#   set.seed(123)
#   model_ann_unscaled <- train(BW ~ ., data = train_cattle, method = "nnet",
#                                trControl = ctrl_cattle, linout = TRUE, trace = FALSE)
#   -> Produced warning "missing values in resampled performance measures"
#      (several tuning combinations returned NaN R-squared -- likely
#      non-convergence on some CV folds). Best result: RMSE 90.18,
#      R-sq 0.347, MAE 69.04 -- poor, attributable to nnet's known
#      sensitivity to predictor scale (traits ranged from ~9 to ~570,
#      very different magnitudes).
#
# FIX applied: added preProcess = c("center","scale"), which standardizes
# all numeric predictors (mean=0, SD=1) using TRAINING SET parameters only
# (caret automatically applies the same transform to the test set at
# prediction time -- no data leakage).
set.seed(123)
model_ann_cattle <- train(BW ~ ., data = train_cattle, method = "nnet",
                           trControl = ctrl_cattle,
                           preProcess = c("center", "scale"),
                           linout = TRUE, trace = FALSE)
model_ann_cattle                  # CV training best (size=3, decay=1e-04): RMSE ~73.4-73.8, R-sq ~0.56, MAE ~55-57
pred_ann_cattle <- predict(model_ann_cattle, newdata = test_cattle)
result_ann <- postResample(pred_ann_cattle, test_cattle$BW)
result_ann                         # Test set: RMSE 69.82, R-sq 0.551, MAE 45.48 -- weakest of all 5 models
# Scaling substantially improved ANN (R-sq 0.347 -> 0.551 on test set) but it
# remains the weakest model overall -- expected, as a simple single-hidden-
# layer network typically needs more than 135 training rows to be
# competitive with tree/kernel methods.


# =================================================================
# STEP 5: PREDICTED VS ACTUAL PLOTS (ALL 5 MODELS)
# =================================================================

png("MLR_predicted_vs_actual.png", width = 800, height = 600)
plot(test_cattle$BW, pred_lm_cattle, main = "MLR: Predicted vs Actual Body Weight",
     xlab = "Actual Body Weight (kg)", ylab = "Predicted Body Weight (kg)",
     pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)
dev.off()

png("RF_predicted_vs_actual.png", width = 800, height = 600)
plot(test_cattle$BW, pred_rf_cattle, main = "Random Forest: Predicted vs Actual Body Weight",
     xlab = "Actual Body Weight (kg)", ylab = "Predicted Body Weight (kg)",
     pch = 19, col = "darkgreen")
abline(0, 1, col = "red", lwd = 2)
dev.off()

png("SVM_predicted_vs_actual.png", width = 800, height = 600)
plot(test_cattle$BW, pred_svm_cattle, main = "SVM: Predicted vs Actual Body Weight",
     xlab = "Actual Body Weight (kg)", ylab = "Predicted Body Weight (kg)",
     pch = 19, col = "purple")
abline(0, 1, col = "red", lwd = 2)
dev.off()

png("XGB_predicted_vs_actual.png", width = 800, height = 600)
plot(test_cattle_xgb$BW, pred_xgb_cattle, main = "XGBoost: Predicted vs Actual Body Weight",
     xlab = "Actual Body Weight (kg)", ylab = "Predicted Body Weight (kg)",
     pch = 19, col = "orange")
abline(0, 1, col = "red", lwd = 2)
dev.off()

png("ANN_predicted_vs_actual.png", width = 800, height = 600)
plot(test_cattle$BW, pred_ann_cattle, main = "ANN: Predicted vs Actual Body Weight",
     xlab = "Actual Body Weight (kg)", ylab = "Predicted Body Weight (kg)",
     pch = 19, col = "brown")
abline(0, 1, col = "red", lwd = 2)
dev.off()

# SHARED FINDING ACROSS ALL 5 MODELS: underprediction of the heaviest animals
# (~400-450kg actual weight), most pronounced in ANN (which shows a nearly
# flat cluster of predictions around 340-350kg for all animals in that
# range). Attributed to sparse training data in that weight range (visible
# in the original bimodal BW histogram, Step 1) rather than a weakness
# specific to any one algorithm -- an important shared limitation for
# Discussion.


# =================================================================
# STEP 6: VARIABLE IMPORTANCE (RANDOM FOREST & XGBOOST)
# =================================================================

varImp_rf <- varImp(model_rf_cattle)
print(varImp_rf)
png("RF_variable_importance.png", width = 800, height = 600)
plot(varImp_rf, main = "Random Forest: Variable Importance")
dev.off()
# Top 5: Heart girth (100), Rump heigth (42.3), Body length (37.7),
# Hind leg length (29.4), Hip height (28.0)

varImp_xgb <- varImp(model_xgb_cattle)
print(varImp_xgb)
png("XGB_variable_importance.png", width = 800, height = 600)
plot(varImp_xgb, main = "XGBoost: Variable Importance")
dev.off()
# Top 5: Body length (100), Heart girth (71.0), Rump heigth (61.6),
# Head length (32.1), Shoulder width (22.4)

# CROSS-MODEL CONSISTENCY: Heart girth, Body length, Rump heigth, Hip height,
# and Hind leg length dominate BOTH RF and XGBoost importance rankings, and
# align with the simple correlation ranking from Step 2 -- a robust,
# multi-method-confirmed finding (correlation + 2 independent ML models),
# not an artifact of one algorithm. This is the central biological finding
# of the study.
#
# NOTABLE EXCEPTION: Rump heigth ranks high in BOTH tree models (RF #2,
# XGBoost #3) despite weak simple correlation (r = 0.379) and the earlier
# VIF flag (15.5). Suggests Rump heigth contributes via non-linear/
# interaction effects with other traits that simple linear correlation
# cannot detect -- a genuinely interesting Discussion point, and a partial
# justification for the earlier decision to retain all traits despite
# multicollinearity concerns.


# =================================================================
# STEP 7: COMBINED MODEL COMPARISON (TEST SET METRICS)
# =================================================================

# Built dynamically from the actual postResample() results computed above
# (not hardcoded), so re-running this script reproduces the table from
# whatever the models actually returned in this session.
comparison_table <- data.frame(
  Model    = c("MLR", "Random Forest", "SVM", "XGBoost", "ANN"),
  RMSE     = c(result_lm["RMSE"],     result_rf["RMSE"],     result_svm["RMSE"],
               result_xgb["RMSE"],    result_ann["RMSE"]),
  Rsquared = c(result_lm["Rsquared"], result_rf["Rsquared"], result_svm["Rsquared"],
               result_xgb["Rsquared"],result_ann["Rsquared"]),
  MAE      = c(result_lm["MAE"],      result_rf["MAE"],      result_svm["MAE"],
               result_xgb["MAE"],     result_ann["MAE"])
)
comparison_table <- comparison_table[order(-comparison_table$Rsquared), ]  # best first
comparison_table

png("Model_comparison_Rsquared.png", width = 800, height = 600)
barplot(comparison_table$Rsquared, names.arg = comparison_table$Model,
        main = "Model Comparison: R-squared (Test Set)", ylab = "R-squared",
        col = c("purple", "darkgreen", "steelblue", "orange", "brown")[
          match(comparison_table$Model, c("SVM","Random Forest","MLR","XGBoost","ANN"))],
        ylim = c(0, 1))
dev.off()

png("Model_comparison_RMSE.png", width = 800, height = 600)
barplot(comparison_table$RMSE, names.arg = comparison_table$Model,
        main = "Model Comparison: RMSE (Test Set)", ylab = "RMSE (kg)",
        col = c("purple", "darkgreen", "steelblue", "orange", "brown")[
          match(comparison_table$Model, c("SVM","Random Forest","MLR","XGBoost","ANN"))])
dev.off()

png("Model_comparison_MAE.png", width = 800, height = 600)
barplot(comparison_table$MAE, names.arg = comparison_table$Model,
        main = "Model Comparison: MAE (Test Set)", ylab = "MAE (kg)",
        col = c("purple", "darkgreen", "steelblue", "orange", "brown")[
          match(comparison_table$Model, c("SVM","Random Forest","MLR","XGBoost","ANN"))])
dev.off()

# NUANCE FOR DISCUSSION: On R-squared and RMSE, SVM is the clear single best
# performer. On MAE specifically, Random Forest (32.54) and SVM (32.69) are
# essentially tied/statistically indistinguishable -- both clearly ahead of
# MLR (40.32) and XGBoost (40.02), with ANN (45.48) weakest throughout. The
# "best model" label can shift slightly depending on which metric is
# emphasized -- worth a nuanced, non-overstated conclusion.


# =================================================================
# FINAL SUMMARY
# =================================================================
# Best overall model (R-squared, RMSE):      SVM (radial kernel)
# Best/tied on MAE:                          Random Forest and SVM (~equivalent)
# Weakest model:                             ANN
# Notable methodological finding:            XGBoost had the best/near-best CV
#                                             training fit but the WORST test-set
#                                             generalization (overfitting) --
#                                             test-set metrics, not CV training
#                                             metrics, should drive model ranking
# Key predictors (consistent across 3 methods -- correlation, RF, XGBoost):
#   Heart girth, Body length, Hip height, Hind leg length, Rump heigth
# Shared limitation across ALL models:       underprediction of heaviest animals
#                                             (~400-450kg), attributable to sparse
#                                             training data in that weight range
# Multicollinearity:                         VIF > 10 for Rump heigth, Rump length,
#                                             Hip height; retained in all models for
#                                             fair comparison, reported as a limitation
