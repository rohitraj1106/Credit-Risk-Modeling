# Loan Default Prediction Using Logistic Regression in R

# Load Libraries
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(caTools)

# Generate synthetic dataset
set.seed(123)
n <- 500
loan_data <- data.frame(
  age = round(runif(n, 18, 65)),
  income = round(rnorm(n, mean = 50000, sd = 15000)),
  loan_amount = round(runif(n, 1000, 25000)),
  credit_score = round(rnorm(n, mean = 650, sd = 70)),
  default = rbinom(n, 1, prob = 0.3)
)

# Check for missing values
sum(is.na(loan_data))
summary(loan_data)

# Visualization
ggplot(loan_data, aes(x = income, fill = as.factor(default))) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.5) +
  labs(title = "Income vs Loan Default", fill = "Default")

# Split the data
set.seed(100)
split <- sample.split(loan_data$default, SplitRatio = 0.7)
train_data <- subset(loan_data, split == TRUE)
test_data <- subset(loan_data, split == FALSE)

# Build logistic regression model
log_model <- glm(default ~ age + income + loan_amount + credit_score,
                 data = train_data,
                 family = binomial)
summary(log_model)

# Predict probabilities
prob_pred <- predict(log_model, newdata = test_data, type = "response")
pred_class <- ifelse(prob_pred > 0.5, 1, 0)

# Confusion Matrix
conf_mat <- confusionMatrix(as.factor(pred_class), as.factor(test_data$default))
print(conf_mat)

# ROC Curve and AUC
roc_obj <- roc(test_data$default, prob_pred)
plot(roc_obj, main = "ROC Curve - Logistic Regression", col = "blue")
auc(roc_obj)
