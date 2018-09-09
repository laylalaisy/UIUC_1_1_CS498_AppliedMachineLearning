import csv
import numpy as np

class_amount = 10

def gaussianNaiveBayes(train_input_x_classified, test_input_x):
    # initialize
    initial_value = 0
    list_length = 10
    train_mean = [initial_value] * list_length
    train_sd = [initial_value] * list_length

    for iter in range(class_amount):
        #train_mean[iter] = np.mean(train_input_x_classified[iter])
        print(train_input_x_classified[iter][0][0])
        break


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
    train_input_x_classified_list = []
    for iter in range(class_amount):
        train_input_x_classified_list.append([])

    train_input_sample_amount = train_input_y.shape[0]

    for iter in range(train_input_sample_amount):
        train_input_x_classified_list[int(train_input_y[iter])].append(train_input_x[iter])

    train_input_x_classified = np.array(train_input_x_classified_list)

    # READ IN TESTING DATA
    with open("./test.csv", "r") as test_input_file:
        test_input_reader = csv.reader(test_input_file)
        # read in testing data
        test_input_data_list = []
        for row in test_input_reader:
            test_input_data_list.append(row)
        # close testing csv file
        test_input_file.close()

    test_input_x = np.array(test_input_data_list)

    # GAUSSIAN
    gaussianNaiveBayes(train_input_x_classified, test_input_x)







