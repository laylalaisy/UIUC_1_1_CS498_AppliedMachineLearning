library(glmnet)
library(klaR)
library(caret)

data(QuickStartExample);

# read in training data and label
data <- as.matrix(read.table("/home/layla/Desktop/cs498/hw7/4/train_data.txt", header=FALSE));
label <- as.matrix(read.table("/home/layla/Desktop/cs498/hw7/4/train_label.txt", header=FALSE));

          
# extract train data by randomly choosing 80% from origin dataset
#train_index <- createDataPartition(y=label, p=0.5, list=FALSE);
# extract training feature vectors and labels
# train_x <- data[train_index, ];
# train_y <- label[train_index, 1];

data(BinomialExample)
fit1 <- glmnet(x, y, family="binomial")
# fit
fit2 <- glmnet(data, label, family="binomial");
plot(fit2, xvar = "dev", label = TRUE);

