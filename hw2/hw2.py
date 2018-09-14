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
    train_sample_amount = train_sample_amount - 1       # sampel_index: 0 - sample_amount
    train_feature_amount = train_feature_amount - 1     # feature_index: 0 - feature_amount

    # extract data of feature and label
    train_input_x = train_input_data[:, :train_feature_amount-1]
    train_input_y = train_input_data[:, train_feature_amount]


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
    train_input_x_rescaled = preprocessing.StandardScaler(train_input_x[:, 1:train_feature_amount-1], with_mean=True, with_std=True)
    test_input_x_rescaled = preprocessing.StandardScaler(test_input_x[:, 1:], with_mean=True, with_std=True)

    print(train_input_data.dtype)