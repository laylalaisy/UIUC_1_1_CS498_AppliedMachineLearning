library(glmnet)
library(klaR)
library(caret)

curpath <- getwd();

# TRAIN
# read in training data and label
filename <- paste(curpath, "/blogData_train.csv", sep="");
train <- as.matrix(read.csv(file=filename, header=FALSE, sep=","));
train_x <- train[,1:280];
train_y <- train[,281];

# fit
cvfit = cv.glmnet(train_x, train_y, family = "poisson", alpha=1);
png(file="1.png");
plot(cvfit);    # plot of the cross-validated deviance of the model against the regularization variable
dev.off();

# predict
print(cvfit$lambda.min);
train_predict_y <- predict(cvfit, newx=train_x, type="response", s = "lambda.min");
# draw plot
png(file="2_train_min.png");
plot(train_y, train_predict_y, main="2_min Train Data", xlab="true values", ylab="predicted values");
dev.off();

# predict
print(cvfit$lambda.1se);
train_predict_y <- predict(cvfit, newx=train_x, type="response", s = "lambda.1se");
# draw plot
png(file="2_train_1se.png");
plot(train_y, train_predict_y, main="2_1se Train Data", xlab="true values", ylab="predicted values");
dev.off();


# TEST
# read in testing data and label
testpath <- paste(curpath, "/data/test/", sep="");
files <- list.files(path=testpath);
num <- length(files);

for (i in 1:num){
  filename <- paste(testpath, files[i], sep="");
  test <- as.matrix(read.csv(file=filename, header=FALSE, sep=","));
  test_x <- test[,1:280];
  test_y <- test[,281];
  
  # predict
  test_predict_y <- predict(cvfit, newx=test_x, type="response", s = "lambda.min");
  # draw plot
  plotname <- paste("3_test_min", i, sep="");
  plotname <- paste(plotname, ".png", sep="");
  png(file=plotname);
  plot(test_y, test_predict_y, main=plotname, xlab="true values", ylab="predicted values");
  dev.off()
  
  # predict
  test_predict_y <- predict(cvfit, newx=test_x, type="response", s = "lambda.1se");
  # draw plot
  plotname <- paste("3_test_1se", i, sep="");
  plotname <- paste(plotname, ".png", sep="");
  png(file=plotname);
  plot(test_y, test_predict_y, main=plotname, xlab="true values", ylab="predicted values");
  dev.off()
}

filename <- paste(curpath, "/blogData_test.csv", sep="");
test <- as.matrix(read.csv(file=filename, header=FALSE, sep=","));
test_x <- test[,1:280];
test_y <- test[,281];
  
# predict
test_predict_y <- predict(cvfit, newx=test_x, type="response", s = "lambda.min");

# draw plot
png(file="3_test.png");
plot(test_y, test_predict_y, main="Test Data", xlab="true values", ylab="predicted values");
dev.off();

