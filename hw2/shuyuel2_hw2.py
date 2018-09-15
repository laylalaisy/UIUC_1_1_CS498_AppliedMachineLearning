import csv
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt

index_x = [0, 2, 4, 10, 11, 12]
feature_amount = 6
train_regularizer = [0.001, 0.01, 0.1, 1]

def stochasticGradientDescent(train_input_x, train_input_y, regularizer, train_sample_amount):

    ## INITIALIZE LIST OF ACCURACY ANF MAGNITUDE
    list_accuracy = []
    list_magnitude = []

    ## INITIALIZE a AND b
    a = np.array([1, 1, 1, 1, 1, 1])
    b = 1.00

    # GRADIENT DESCENT
    ## TRAIN
    amount_epoch = 50
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
        train_input_x_validation = np.delete(train_input_x_epoch, index_step, axis=0)
        train_input_y_validation = np.delete(train_input_y_epoch, index_step, axis=0)

        # renew a and b
        for iter_step in range(amount_step):
            xi = train_input_x_step[iter_step, :]
            yi = train_input_y_step[iter_step]
            gi = yi * ((a).dot(xi) + b)

            if (gi >= 1):
                a = a - step_length * regularizer * a
            else:
                a = a - step_length * (regularizer * a - yi * xi)
                b = b + step_length * yi

            # EVERY 30 STEPS
            if(iter_step % 30 == 0):
                # predict label of training set and get accuracy
                correct_amount = 0
                for iter_y in range(amount_step):
                    if train_input_y_step[iter_y] * ((a).dot(train_input_x_step[iter_y, :]) + b) > 0:
                        correct_amount = correct_amount + 1
                accuracy = float(correct_amount / amount_step)
                list_accuracy.append(accuracy)

                # get magnitude
                magnitude = (a).dot(a.T)
                list_magnitude.append(magnitude)

    # predict label of validation set
    correct_amount = 0
    for iter_y in range(amount_validation):
        if train_input_y_validation[iter_y] * ((a).dot(train_input_x_validation[iter_y, :]) + b) > 0:
            correct_amount = correct_amount + 1

    # calculate accuracy
    accuracy_validation = correct_amount / amount_validation

    return a, b, accuracy_validation, list_accuracy, list_magnitude

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

    # classify training labels
    train_input_y = train_input_data[:, train_feature_amount-1]
    for iter_y in range(0, train_sample_amount):
        if train_input_y[iter_y] == ' <=50K':   # <=50K
            train_input_y[iter_y] = -1
        else:                                   # >50K
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


    ## VALIDATION TO GET BEST REGULARIZER
    best_accuracy = 0
    best_a = np.array([1, 1, 1, 1, 1, 1])
    best_b = 1.00
    best_regularizer = 0
    accuray = []
    magnitude = []
    for regularizer in train_regularizer:
        [current_a, current_b, current_accuracy, list_accuracy, list_magnitude] = stochasticGradientDescent(train_input_x_rescaled, train_input_y, regularizer, train_sample_amount)
        if current_accuracy > best_accuracy:
            best_a = current_a
            best_b = current_b
            best_accuracy = current_accuracy
            best_regularizer = regularizer

        accuray.append(list_accuracy)
        magnitude.append(list_magnitude)
    print("best regularizer:", best_regularizer)


    ## TEST
    test_output_y = []
    for iter_y in range(test_sample_amount):
        if best_a.dot(test_input_x_rescaled[iter_y, :]) + best_b > 0:
            test_output_y.append('>50K')
        else:
            test_output_y.append('<=50K')
    test_output_y = np.array(test_output_y)
    writeCsvFile("hw2.csv", test_output_y)


    ## DRAW
    image_a = plt.figure()
    for iter in range(len(train_regularizer)):
        y = accuray[iter]
        x = np.arange(len(y))

        image_accuracy = image_a.add_subplot(111)
        image_accuracy.plot(x, y)

    image_accuracy.legend(train_regularizer)
    image_accuracy.set_xlabel('Steps')
    image_accuracy.set_ylabel('Accuracy')

    image_a.savefig('accuracy.png')
    image_a.show()

    image_m = plt.figure()
    for iter in range(len(train_regularizer)):
        y = magnitude[iter]
        x = np.arange(len(y))

        image_magnitude = image_m.add_subplot(111)
        image_magnitude.plot(x, y)

    image_magnitude.legend(train_regularizer)
    image_magnitude.set_xlabel('Steps')
    image_magnitude.set_ylabel('Magnitude')

    image_m.savefig('magnitude.png')
    image_m.show()