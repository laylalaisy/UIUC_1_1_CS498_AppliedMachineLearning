import csv
from sklearn import preprocessing
import numpy as np

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

    amount_epoch = 5
    amount_step = 300
    amount_validation = 50
    for i in range(amount_epoch):
        index_epoch = np.random.choice(train_sample_amount, size=amount_step + amount_validation, replace=False)
        train_input_x_epoch = train_input_x[index_epoch, :]
        index_step = np.random.choice(train_input_x_epoch.shape[0], size=amount_step, replace=False)
        train_input_x_step = train_input_x_epoch[index_step, :]
        index_validation = np.delete(train_input_x_epoch, index_step, axis=0)

