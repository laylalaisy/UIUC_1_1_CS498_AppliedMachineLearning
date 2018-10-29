library(glmnet)
library(klaR)
library(caret)

curpath <- getwd();

# read in training data and label
datafile <- paste(curpath, "/train_data.txt", sep="");
labelfile <- paste(curpath, "/train_label.txt", sep="");
data <- t(as.matrix(read.table(datafile, header=FALSE)));
label <- as.matrix(read.table(labelfile, header=FALSE));

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
png(file="cvfit.png");
plot(cvfit);
dev.off();

# predict
print(cvfit$lambda.min);
test_predict_y <- predict(cvfit, newx=test_x , type="class", s = "lambda.min");
# accuracy
accuracy_table <- table(test_predict_y == test_y);
print(accuracy_table);
accuracy <- accuracy_table[2]/(accuracy_table[1]+accuracy_table[2])
print(accuracy);

# predict
print(cvfit$lambda.1se);
test_predict_y <- predict(cvfit, newx=test_x , type="class", s = "lambda.1se");
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