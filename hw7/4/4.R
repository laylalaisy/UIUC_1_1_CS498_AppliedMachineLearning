library(glmnet)
library(klaR)
library(caret)

data(QuickStartExample);

# read in training data and label
data <- t(as.matrix(read.table("/home/layla/Desktop/cs498/hw7/4/train_data.txt", header=FALSE)));
label <- as.matrix(read.table("/home/layla/Desktop/cs498/hw7/4/train_label.txt", header=FALSE));

# extract train data by randomly choosing 80% from origin dataset
train_index <- createDataPartition(y=label, p=0.8, list=FALSE);
# extract training feature vectors and labels
train_x <- data[train_index, ];
train_y <- label[train_index, 1];

# extract testing feature vectors and labels
test_x <- data[-train_index, ];
test_y <- label[-train_index, 1];

# fit: cross-validation
cvfit = cv.glmnet(train_x, train_y, family = "binomial", type.measure = "class");
plot(cvfit);

# predict
print(cvfit$lambda.min);
test_predict_y <- predict(cvfit, newx=test_x , type="class", s = "lambda.min");

# accuracy
accuracy_table <- table(test_predict_y == test_y);
print(accuracy_table);
accuracy <- accuracy_table[2]/(accuracy_table[1]+accuracy_table[2])
print(accuracy);

# baseline
tissue_table <- table(test_y == 0);
print(tissue_table);
accuracy_baseline <- max(tissue_table[1], tissue_table[2])/(tissue_table[1]+tissue_table[2])
print(accuracy_baseline);