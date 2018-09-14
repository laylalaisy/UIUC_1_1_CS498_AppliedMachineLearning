import csv
from sklearn import preprocessing
import numpy as np

def stochasticGradientDescent(trainX, trainY, validationX, validationY, regularizer, step_length):
    [train_sample_amount, train_feature_amount] = trainX.shape


    ## INITIALIZE a AND b
    a = np.ones([train_feature_amount]).astype(float)
    b = 1

    #for step in range(train_input_x):
    print(a)

if __name__ == "__main__":

    ## READ IN TRAINING DATA
    with open("./Data/train.data", "r") as train_input_file:
        train_input_reader = csv.reader(train_input_file)
        train_input_data_list = []
        for row in train_input_reader:
            train_input_data_list.append(row)
        train_input_file.close()

    # change input data from list to array
    train_input_data = np.array(train_input_data_list)

    # get training data set size
    [train_sample_amount, train_feature_amount] = train_input_data.shape

    # extract data of feature and label
    train_input_x = train_input_data[:, :train_feature_amount-2]
    train_input_y = train_input_data[:, train_feature_amount-1]


    ## READ IN TESTING DATA
    with open("./Data/test.data", "r") as test_input_file:
        test_input_reader = csv.reader(test_input_file)
        test_input_data_list = []
        for row in test_input_reader:
            test_input_data_list.append(row)
        test_input_file.close()

    # change input data from list to array
    test_input_data = np.array(test_input_data_list)

    # extract data of feature
    test_input_x = test_input_data[:, :]

    ## RESCALE
    # Scale these variables so that each has unit variance
    # And subtract the mean so that each has zero mean
    train_input_x_rescaled = preprocessing.StandardScaler(train_input_x[:, 1:train_feature_amount-2], with_mean=True, with_std=True)
    test_input_x_rescaled = preprocessing.StandardScaler(test_input_x[:, 1:], with_mean=True, with_std=True)

    ## TRAIN
    amount_epoch = 1
    amount_step = 300
    amount_validation = 50
    for iter_epoch in range(amount_epoch):
        # data of the whole epoch
        index_epoch = np.random.choice(train_sample_amount, size=amount_step + amount_validation, replace=False)
        train_input_x_epoch = train_input_x[index_epoch, :]
        train_input_y_epoch = train_input_y[index_epoch]

        # training data and validation data
        index_step = np.random.choice(train_input_x_epoch.shape[0], size=amount_step, replace=False)
        train_input_x_step = train_input_x_epoch[index_step, :]
        train_input_y_step = train_input_y_epoch[index_step]
        train_input_x__validation = np.delete(train_input_x_epoch, index_step, axis=0)
        train_input_y__validation = np.delete(train_input_y_epoch, index_step, axis=0)

        # step_length
        step_length = 1.0 / ((0.01 * iter_epoch) + 50)
        stochasticGradientDescent(train_input_x_step, train_input_y_step, train_input_x__validation, train_input_y__validation, 1, step_length)

