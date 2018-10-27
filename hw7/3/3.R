library(glmnet)

# read in training data and label
train <- as.matrix(read.csv(file="/home/layla/Desktop/cs498/hw7/3/data/train/blogData_train.csv", header=FALSE, sep=","));
train_x <- train[,1:280];
train_y <- train[,281];

# fit
cvfit = cv.glmnet(train_x, train_y, family = "poisson", alpha=1);
plot(cvfit);

# predict
print(cvfit$lambda.min);
train_predict_y <- predict(cvfit, newx=train_x, type="response", s = "lambda.min");
plot(train_y, train_predict_y, main="Train Data", xlab="true values", ylab="predicted values")