import csv
from sklearn import preprocessing
import numpy as np

index_x = [0, 2, 4, 10, 11, 12]
feature_amount = 6


def stochasticGradientDescent(train_input_x, train_input_y, regularizer, train_sample_amount):

    ## INITIALIZE a AND b
    a = np.matrix('1,1,1,1,1,1')
    b = 1.00

    # GRADIENT DESCENT
    ## TRAIN
    amount_epoch = 500
    amount_step = 300
    amount_validation = 50

    for iter_epoch in range(amount_epoch):
        # step_length
        step_length = 1.0 / ((0.01 * iter_epoch) + 50)

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

        for iter_step in range(amount_step):
            xi = train_input_x_step[iter_step, :]
            yi = train_input_y_step[iter_step]
            gi = yi * ((xi).dot(a.T) + b)

            if (gi >= 1):
                a = a - step_length * regularizer * a
            else:
                a = a - step_length * (regularizer * a - yi * xi)
                b = b + step_length * yi


    return a, b

# output result in csv file
def writeCsvFile(filename, test_output_y):
    with open(filename, "w") as test_output_file:
        test_output_writer = csv.writer(test_output_file)

        # write header
        fileHeader = ["Example", "Label"]
        test_output_writer.writerow(fileHeader)

        # write content
        test_sample_amount = test_output_y.shape[0]
        content = []
        for iter in range(test_sample_amount):
            string_index = "'" + str(iter) + "'"
            content.append([string_index, test_output_y[iter]])
        test_output_writer.writerows(content)


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
    train_sample_amount = train_input_data.shape[0]
    train_feature_amount = train_input_data.shape[1]

    # extract data of feature and label
    train_input_x = train_input_data[:, index_x]
    train_input_x = np.array(train_input_x)

    train_input_y = train_input_data[:, train_feature_amount-1]
    for iter_y in range(0, train_sample_amount):
        if train_input_y[iter_y] == ' <=50K':
            train_input_y[iter_y] = -1
        else:
            train_input_y[iter_y] = 1
    train_input_y = np.array(train_input_y).astype(int)

    ## READ IN TESTING DATA
    with open("./Data/test.data", "r") as test_input_file:
        test_input_reader = csv.reader(test_input_file)
        test_input_data_list = []
        for row in test_input_reader:
            test_input_data_list.append(row)
        test_input_file.close()

    # change input data from list to array
    test_input_data = np.array(test_input_data_list)
    test_sample_amount = test_input_data.shape[0]

    # extract data of feature
    test_input_x = test_input_data[:, index_x]


    ## RESCALE
    # Scale these variables so that each has unit variance
    # And subtract the mean so that each has zero mean
    train_input_x_rescaled = preprocessing.scale(train_input_x, axis=0, with_mean=True, with_std=True)
    np.array(train_input_x_rescaled).astype(float)
    test_input_x_rescaled = preprocessing.scale(test_input_x, axis=0, with_mean=True, with_std=True)
    np.array(test_input_x_rescaled).astype(float)

    print(train_input_x_rescaled.shape)

    [a, b] = stochasticGradientDescent(train_input_x_rescaled, train_input_y, 0.001, train_sample_amount)

    test_output_y = []
    for iter_y in range(test_sample_amount):
        if (test_input_x_rescaled[iter_y, :]).dot(a.T) + b > 0:
            test_output_y.append('>50K')
            print("OMG")
        else:
            test_output_y.append('<=50K')
            print("FUCK")

    test_output_y = np.array(test_output_y)

    writeCsvFile("demo.csv", test_output_y)