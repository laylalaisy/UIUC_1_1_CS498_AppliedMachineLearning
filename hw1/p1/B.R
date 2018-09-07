# Build a simple naive Bayes classifier to classify this data set.

# load library
library(klaR)
library(caret)

# set current work path
setwd('.')

# read in csv file
input_data <- read.csv('pima-indians-diabetes.csv', header=FALSE)

# split the inputdata
input_x <- input_data[,-c(9)]  # features
input_y <- input_data[,9]      # labels

# deal with attribute with zero value
for (index_feature in c(3, 5, 6, 8))
{
  index_NA <- input_x[, index_feature] == 0
  input_x[index_NA, index_feature]=NA
}


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

  # get positive index
  positive_index <- train_y == 1

  # get positive & negative feature data
  train_x_positive <- train_x[positive_index, ]
  train_x_negative <- train_x[!positive_index,]

  # use a normal distribution to model distributions
  # calculate mean and standard deviation
  train_x_positive_mean <- sapply(train_x_positive, mean, na.rm=TRUE)
  train_x_negative_mean <- sapply(train_x_negative, mean, na.rm=TRUE)

  train_x_positive_sd <- sapply(train_x_positive, sd, na.rm=TRUE)
  train_x_negative_sd <- sapply(train_x_negative, sd, na.rm=TRUE)

  # calculate log probability
  train_x_positive_offset <- t(t(train_x) - train_x_positive_mean)
  train_x_positive_scale <- t(t(train_x_positive_offset) / train_x_positive_sd)
  train_x_positive_square <- apply(train_x_positive_scale, c(1, 2), function(x)x^2)
  train_x_positive_log_prob <- -(1/2)*rowSums(train_x_positive_square, na.rm=TRUE) - sum(log(train_x_positive_sd))

  train_x_negative_offset <- t(t(train_x) - train_x_negative_mean)
  train_x_negative_scale <- t(t(train_x_negative_offset) / train_x_negative_sd)
  train_x_negative_square <- apply(train_x_negative_scale, c(1, 2), function(x)x^2)
  train_x_negative_log_prob <- -(1/2)*rowSums(train_x_negative_square, na.rm=TRUE) - sum(log(train_x_negative_sd))

  # record percentage guessed correctly in scores array
  train_predict_y <- train_x_positive_log_prob > train_x_negative_log_prob
  train_correct_account <- train_predict_y == train_y
  train_accuracy[iter] <- sum(train_correct_account)/(sum(train_correct_account)+sum(!train_correct_account))

  # TEST
  # extract feature vectors and labels
  test_x <- input_x[-train_index, ]
  test_y <- input_y[-train_index]

  # calculate log probability
  test_x_positive_offset <- t(t(test_x) - train_x_positive_mean)
  test_x_positive_scale <- t(t(test_x_positive_offset) / train_x_positive_sd)
  test_x_positive_square <- apply(test_x_positive_scale, c(1, 2), function(x)x^2)
  test_x_positive_log_prob <- -(1/2)*rowSums(test_x_positive_square, na.rm=TRUE) - sum(log(train_x_positive_sd))

  test_x_negative_offset <- t(t(test_x) - train_x_negative_mean)
  test_x_negative_scale <- t(t(test_x_negative_offset) / train_x_negative_sd)
  test_x_negative_square <- apply(test_x_negative_scale, c(1, 2), function(x)x^2)
  test_x_negative_log_prob <- -(1/2)*rowSums(test_x_negative_square, na.rm=TRUE) - sum(log(train_x_negative_sd))

  # record percentage guessed correctly in scores array
  test_predict_y <- test_x_positive_log_prob > test_x_negative_log_prob
  test_correct_account <- test_predict_y == test_y
  test_accuracy[iter] <- sum(test_correct_account)/(sum(test_correct_account)+sum(!test_correct_account))
}

result <- mean(test_accuracy)
print(result)


