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

  # extract feature vectors and labels
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
  train_x_positive_scaled <- t(t(train_x_positive_offset) / train_x_positive_sd)
  train_x_positive_log_prob <- -(1/2)*apply(train_x_positive_scaled, 1, function(x){sum(x^2)}) - sum(log(train_x_positive_sd))

  print(dim(train_x_positive_log_prob))
  train_x_negative_offset <- t(t(train_x) - train_x_negative_mean)
  train_x_negative_scaled <- t(t(train_x_negative_offset) / train_x_negative_sd)
  train_x_negative_log_prob <- -(1/2)*apply(train_x_negative_scaled, 1, function(x){sum(x^2)}) - sum(log(train_x_negative_sd))

   # record percentage guessed correctly in scores array
   predict_y <- train_x_positive_log_prob > train_x_negative_log_prob
   print(predict_y)
   train_correct_account <- predict_y == train_y
   train_accuracy[iter]<-sum(train_correct_account)/(sum(train_correct_account)+sum(!train_correct_account))

   print(train_accuracy[iter])

   # ## evaluating - testing data
   #
   # # extract *testing* feature vectors and labels
   # x_vec_test<-x_vec[-wtd, ]
   # y_vec_test<-y_vec[-wtd]
   #
   # # Solve for log probability that each feature vector corresponds to a positive label
   # pos_test_offset<-t(t(x_vec_test)-pos_mean)
   # pos_test_scaled<-t(t(pos_test_offset)/pos_sd)
   # pos_test_log_prob<--(1/2)*rowSums(apply(pos_test_scaled, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(pos_sd))
   #
   # # Solve for log probability that each feature vector corresponds to a negative label
   # neg_test_offset<-t(t(x_vec_test)-neg_mean)
   # neg_test_scaled<-t(t(neg_test_offset)/neg_sd)
   # neg_test_log_prob<--(1/2)*rowSums(apply(neg_test_scaled, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(neg_sd))
   #
   # # record percentage guessed correctly in scores array
   # guesses_test<-pos_test_log_prob>neg_test_log_prob
   # num_correct_test<-guesses_test==y_vec_test
   # test_score[iter]<-sum(num_correct_test)/(sum(num_correct_test)+sum(!num_correct_test))
 }

# print(test_score)

