import csv
import numpy as np



if __name__ == "__main__":

    # read in train csv file
    with open("./train.csv", "r") as train_input_file:
        train_input_reader = csv.reader(train_input_file)
        # read in train data
        train_input_data_list = []
        for row in train_input_reader:
            train_input_data_list.append(row)
        # close train csv file
        train_input_file.close()

    # change input data from list to array
    train_input_data = np.array(train_input_data_list)

    # extract data of feature and label
    train_input_x = train_input_data[1:, 2:]
    train_input_y = train_input_data[1:, 1:2]

    # read in test csv file
    with open("./test.csv", "r") as test_input_file:
        test_input_reader = csv.reader(test_input_file)
        # read in test data
        test_input_data_list = []
        for row in test_input_reader:
            test_input_data_list.append(row)
        # close test csv file
        test_input_file.close()

    # change input data from list to array
    test_input_x = np.array(test_input_data_list)






