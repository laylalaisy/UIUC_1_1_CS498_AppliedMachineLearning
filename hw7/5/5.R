library(glmnet)
library(klaR)
library(caret)

# read in training data and label
data <- read.csv("./Crusio1.csv", header=TRUE, na.strings=c(""))
data_1 <- na.omit(data[,c(2,4:41)])

# split train-test
wtd <- createDataPartition(y=data_1[,1], p=.85, list=FALSE)

# data processing
data_train <- data_1[wtd,]
train_x_vec <- as.matrix(data_train[,-c(1)])
train_y_vec <- as.factor(data_train[,c(1)])
data_test <- data_1[-wtd,]
test_x_vec <- as.matrix(data_test[,-c(1)])

# train model
cvfit = cv.glmnet(train_x_vec, train_y_vec, family = "binomial", type.measure = "class")
plot(cvfit)

# predict
fitted.results <- predict(cvfit, newx = test_x_vec, s = "lambda.min", type = "class")

baseline <- mean(data_test$sex == 'm')
misClasificError <- mean(fitted.results != data_test$sex)
print(paste('Accuracy',1-misClasificError))
print(paste('Baseline',baseline))
print(paste('Minlambda',cvfit$lambda.min))


# multinomial

data_2 <- na.omit(data[,c(1,4:41)])
data_2_new <- data_2[data_2$strain %in% names(which(table(data_2$strain) > 10)), ]

# split train-test
wtd <- createDataPartition(y=data_2_new[,1], p=.85, list=FALSE)

# data processing
data_train_2 <- data_2_new[wtd,]
train_x_vec_2 <- as.matrix(data_train_2[,-c(1)])
train_y_vec_2 <- as.matrix(data_train_2[,c(1)])
data_test_2 <- data_2_new[-wtd,]
test_x_vec_2 <- as.matrix(data_test_2[,-c(1)])

# train model
train_y_vec_2_new <- as.factor(train_y_vec_2)
cvfit=cv.glmnet(train_x_vec_2, train_y_vec_2_new, family="multinomial", type.measure = "class")
plot(cvfit)

# predict
fitted.results <- predict(cvfit, newx = test_x_vec_2, s = "lambda.min", type = "class")

baseline <- mean(data_test_2$strain == 'BXD100')
misClasificError <- mean(fitted.results != data_test_2$strain)
print(paste('Accuracy',1-misClasificError))
print(paste('Baseline',baseline))
print(paste('Minlambda',cvfit$lambda.min))
