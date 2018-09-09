import csv
import numpy as np

# def gaussianNaiveBayes(train_input_x, train_input_y, test_input_x):


if __name__ == "__main__":

    # READ IN TRAINING DATA
    with open("./train.csv", "r") as train_input_file:
        train_input_reader = csv.reader(train_input_file)
        # read in training data
        train_input_data_list = []
        for row in train_input_reader:
            train_input_data_list.append(row)
        # close training csv file
        train_input_file.close()

    # change input data from list to array
    train_input_data = np.array(train_input_data_list)

    # extract data of feature and label
    train_input_x = train_input_data[1:, 2:]
    train_input_y = train_input_data[1:, 1:2]


    # STORE CLASSIFIED TRAINING DATA
    test_input_x = np.array(test_input_data_list)

    train_input_x_classified = []
    for iter in range(10):
        train_input_x_classified.append([])

    train_input_sample_amount = train_input_y.shape[0]

    for iter in range(train_input_sample_amount):
        train_input_x_classified[int(train_input_y[iter])].append(train_input_x[iter])

    print(train_input_x_classified)


    # READ IN TESTING DATA
    with open("./test.csv", "r") as test_input_file:
        test_input_reader = csv.reader(test_input_file)
        # read in testing data
        test_input_data_list = []
        for row in test_input_reader:
            test_input_data_list.append(row)
        # close testing csv file
        test_input_file.close()







