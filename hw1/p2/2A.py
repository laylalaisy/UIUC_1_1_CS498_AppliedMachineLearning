import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

class_amount = 10
origin_scale = 28

def featureShape(input_x):
    input_x_reshape = []
    sample_amount = input_x.shape[0]
    for iter in range(sample_amount):
        input_x_reshape.append(np.reshape(input_x[iter], [28, 28]))
    input_x_reshape = np.array(input_x_reshape)
    input_x_reshape = np.array(input_x_reshape).astype(np.float)
    return input_x_reshape

def stretchedBoundingBox(input_x_reshape):
    input_x_stretched = []
    sample_amount = len(input_x_reshape)
    for iter in range(sample_amount):
        row_left = origin_scale
        row_right = 0
        col_down = origin_scale
        col_up = 0
        for row in range(origin_scale):
            for col in range(origin_scale):
                if(input_x_reshape[iter][row][col]>0):
                    if(row<row_left):
                        row_left=row
                    if(row>row_right):
                        row_right = row
                    if(col<col_down):
                        col_down = col
                    if(col>col_up):
                        col_up=col
        row_new = []
        for row in range(col_down, col_up):
            col_new = []
            for col in range(row_left, row_right):
                col_new.append(input_x_reshape[iter][row][col])
            row_new.append(col_new)
        row_new = np.array(row_new)
        row_new = np.array(row_new).astype(np.float)
        input_x_stretched.append(np.resize(row_new, (20, 20)))

    return input_x_stretched

def gaussianNaiveBayes(train_input_x, train_input_y, test_input_x):
    # # initialize
    # initial_value = 0
    # list_length = 10
    # train_mean = [initial_value] * list_length
    # train_sd = [initial_value] * list_length
    #
    # for iter in range(class_amount):
    #     # train_mean[iter] = np.mean(train_input_x_classified[iter])
    #     print(train_input_x_classified[iter][0][0])
    #     break
    classifier = GaussianNB()
    classifier.fit(train_input_x, train_input_y.ravel())

    test_output_y = classifier.predict(test_input_x)
    return test_output_y

def bernoulliNaiveBayes(train_input_x, train_input_y, test_input_x):
    classifier = BernoulliNB()
    classifier.fit(train_input_x, train_input_y.ravel())

    test_output_y = classifier.predict(test_input_x)
    return test_output_y

def writeCsvFile(filename, test_output_y):
    with open(filename, "w") as test_output_file:
        test_output_writer = csv.writer(test_output_file)

        fileHeader = ["ImageId", "Label"]
        test_output_writer.writerow(fileHeader)

        test_sample_amount = test_output_y.shape[0]
        content = []
        for iter in range(test_sample_amount):
            content.append([iter, int(test_output_y[iter])])
        test_output_writer.writerows(content)



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
    train_input_x = np.array(train_input_x).astype(np.float)
    train_input_y = train_input_data[1:, 1:2]
    train_input_y = np.array(train_input_y).astype(np.float)

    # # STORE CLASSIFIED TRAINING DATA
    # train_input_x_classified_list = []
    # for iter in range(class_amount):
    #     train_input_x_classified_list.append([])
    #
    # train_input_sample_amount = train_input_y.shape[0]
    #
    # for iter in range(train_input_sample_amount):
    #     train_input_x_classified_list[int(train_input_y[iter])].append(train_input_x[iter])
    # train_input_x_classified = np.array(train_input_x_classified_list)

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
    test_input_x = np.array(test_input_x).astype(np.float)

    # 1. GAUSSIAN + UNTOUCHED
    # test_output_y = gaussianNaiveBayes(train_input_x, train_input_y, test_input_x)
    # writeCsvFile("shuyuel2_1.csv", test_output_y)

    # 3. BERNOULLI + UNTOUCHED
    # test_output_y = bernoulliNaiveBayes(train_input_x, train_input_y, test_input_x)
    # writeCsvFile("shuyuel2_3.csv", test_output_y)

    # RESHAPE FEATURE VECTORS
    train_input_x_reshape = featureShape(train_input_x)
    test_input_x_reshape = featureShape(test_input_x)

    # STRETCHED BOUNDING BOX
    train_input_x_stretched = stretchedBoundingBox(train_input_x_reshape)
    test_input_x_stretched = stretchedBoundingBox(test_input_x_reshape)









