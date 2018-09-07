# Build a SVM classifier to classify this data set.

# load library
library(klaR)
library(caret)

# set current work path
setwd('./')

# read in csv file
input_data <- read.csv('pima-indians-diabetes.csv', header=FALSE)

# split the inputdata
input_x <- input_data[,-c(9)]  # features
input_y <- input_data[,9]      # labels

# number of times
iters = 10

# initialize accuracy
train_accuracy <- array(dim=iters)
test_accuracy <- array(dim=iters)

# run 10 times
for (iter in 1:iters)
{
  # TRAIN
  # extract train data by randomly assigning 80% of the data to train
  train_index <- createDataPartition(y=input_y, p=.8, list=FALSE)
  
  # extract training feature vectors and labels
  train_x <- input_x[train_index, ]
  train_y <- input_y[train_index]
  
  # svmlight (features, labels, pathsvm)
  svm = svmlight(train_x, train_y, pathsvm='./svm_light_linux64/')
  
  # TEST
  # extract feature vectors and labels
  test_x <- input_x[-train_index, ]
  test_y <- input_y[-train_index]
  
  predict_y = predict(svm, test_x)
  
  # calculate test accuracy
  test_correct_account <- predict_y$class == test_y
  test_accuracy[iter] <- sum(test_correct_account)/(sum(test_correct_account)+sum(!test_correct_account))
} 
  
result <- mean(test_accuracy)
print(result)
  