import csv
import numpy as np
import skimage.transform
from PIL import Image

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

# class amount: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
class_amount = 10

# original image scale: 28 * 28
origin_scale = 28

# reshape input_x / feature vectore: [1, 748] -> [28, 28]
def featureShape(input_x):
    input_x_reshape = []
    sample_amount = input_x.shape[0]
    for iter in range(sample_amount):
        input_x_reshape.append(np.reshape(input_x[iter], [28, 28]))
    input_x_reshape = np.array(input_x_reshape)
    input_x_reshape = np.array(input_x_reshape).astype(np.float)
    return input_x_reshape

# get stretchedBoundingBox: reshape input_x / feature vectore: [28, 28] -> [20, 20]
def stretchedBoundingBox(input_x_reshape):
    input_x_stretched = []

    sample_amount = len(input_x_reshape)
    for iter in range(sample_amount):
        row_left = origin_scale
        row_right = 0
        col_down = 0
        col_up = origin_scale

        # get scale of digit
        for row in range(origin_scale):
            for col in range(origin_scale):
                if(input_x_reshape[iter][row][col]>0):
                    if(row<row_left):
                        row_left=row
                    if(row>row_right):
                        row_right = row
                    if(col<col_up):
                        col_up = col
                    if(col>col_down):
                        col_down = col

        # rebuild image
        row_new = []
        for row in range(col_up, col_down):
            col_new = []
            for col in range(row_left, row_right):
                col_new.append(input_x_reshape[iter][row][col])
            row_new.append(col_new)

        # resize image
        row_new = np.array(row_new).astype(np.float)
        input_x_stretched.append(skimage.transform.resize(np.array(row_new), (20, 20), preserve_range=True).ravel())

    return input_x_stretched

# output result in csv file
def writeCsvFile(filename, test_output_y):
    with open(filename, "w") as test_output_file:
        test_output_writer = csv.writer(test_output_file)

        # write header
        fileHeader = ["ImageId", "Label"]
        test_output_writer.writerow(fileHeader)

        # write content
        test_sample_amount = test_output_y.shape[0]
        content = []
        for iter in range(test_sample_amount):
            content.append([iter, int(test_output_y[iter])])
        test_output_writer.writerows(content)

# get mean image of result
def meanImage(test_input_x, test_output_y, scale, image_name):
    # classified feature vector in each class
    test_input_x_classified = []
    test_sample_amount = test_output_y.shape[0]
    for iter in range(class_amount):
        test_input_x_classified.append([])

    test_sample_amount = test_output_y.shape[0]
    for iter in range(test_sample_amount):
        lable = int(test_output_y[iter])
        test_input_x_classified[lable].append(test_input_x[iter])

    # transfer from list to np.array
    meanPixel = []
    for iter in range(class_amount):
        test_input_x_classified[iter] = np.array(test_input_x_classified[iter]).astype(float)
        meanPixel.append(np.mean(test_input_x_classified[iter], axis=0))

    # calculate mean of each class
    for iter in range(class_amount):
        meanPixel[iter] = np.reshape(meanPixel[iter], [scale, scale])
        image = Image.fromarray(meanPixel[iter])
        # image.show()
        image = image.convert("L")
        image_file_name = image_name + str(iter) + ".png"
        image.save(image_file_name)

    return test_input_x_classified

# Gaussian Naive Bayes Classifier
def gaussianNaiveBayes(train_input_x, train_input_y, test_input_x):
    classifier = GaussianNB()
    classifier.fit(train_input_x, train_input_y.ravel())

    test_output_y = classifier.predict(test_input_x)
    return test_output_y

# Bernoulli Naive Bayes Classifier
def bernoulliNaiveBayes(train_input_x, train_input_y, test_input_x):
    classifier = BernoulliNB()
    classifier.fit(train_input_x, train_input_y.ravel())

    test_output_y = classifier.predict(test_input_x)
    return test_output_y

# Random Forest Classifier
def randomForest(num_tree, max_depth, train_input_x, train_input_y, test_input_x):
    classifier = RandomForestClassifier(n_estimators=num_tree, criterion='entropy', max_depth=max_depth)
    classifier.fit(train_input_x, train_input_y)

    test_output_y = classifier.predict(test_input_x)
    return test_output_y


if __name__ == "__main__":
    ## READ IN TRAINING DATA
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


    ## READ IN TESTING DATA
    with open("./test.csv", "r") as test_input_file:
        test_input_reader = csv.reader(test_input_file)
        # read in testing data
        test_input_data_list = []
        for row in test_input_reader:
            test_input_data_list.append(row)
        # close testing csv file
        test_input_file.close()

    test_input_x = np.array(test_input_data_list).astype(np.float)


    ## RESHAPE FEATURE VECTORS
    train_input_x_reshape = featureShape(train_input_x)
    test_input_x_reshape = featureShape(test_input_x)

    ## GET STRETCHED BOUNDING BOX
    train_input_x_stretched = stretchedBoundingBox(train_input_x_reshape)
    test_input_x_stretched = stretchedBoundingBox(test_input_x_reshape)


    ## EXECUTE
    # 1. GAUSSIAN + UNTOUCHED
    test_output_y = gaussianNaiveBayes(train_input_x, train_input_y, test_input_x)
    writeCsvFile("shuyuel2_1.csv", test_output_y)

    test_output_y = np.array(test_output_y).astype(float)
    meanImage(test_input_x, test_output_y, 28, "shuyuel2_1_")

    # 2. GAUSSIAN + STRETCHED
    test_output_y = gaussianNaiveBayes(train_input_x_stretched, train_input_y, test_input_x_stretched)
    writeCsvFile("shuyuel2_2.csv", test_output_y)

    test_output_y = np.array(test_output_y).astype(float)
    meanImage(test_input_x_stretched, test_output_y, 20, "shuyuel2_2_")

    # 3. BERNOULLI + UNTOUCHED
    test_output_y = bernoulliNaiveBayes(train_input_x, train_input_y, test_input_x)
    writeCsvFile("shuyuel2_3.csv", test_output_y)

    test_output_y = np.array(test_output_y).astype(float)
    meanImage(test_input_x, test_output_y, 28, "shuyuel2_3_")

    # 4. BERNOULLI + STRETCHED
    test_output_y = bernoulliNaiveBayes(train_input_x_stretched, train_input_y, test_input_x_stretched)
    writeCsvFile("shuyuel2_4.csv", test_output_y)

    test_output_y = np.array(test_output_y).astype(float)
    meanImage(test_input_x_stretched, test_output_y, 20, "shuyuel2_4_")

    # 5. 10 TREES + 4 DEPTH + UNTOUCHED
    test_output_y = randomForest(10, 4, train_input_x, train_input_y, test_input_x)
    writeCsvFile("shuyuel2_5.csv", test_output_y)

    # 6. 10 TREES + 4 DEPTH + STRETCHED
    test_output_y = randomForest(10, 4, train_input_x_stretched, train_input_y, test_input_x_stretched)
    writeCsvFile("shuyuel2_6.csv", test_output_y)

    # 7. 10 TREES + 16 DEPTH + UNTOUCHED
    test_output_y = randomForest(10, 16, train_input_x, train_input_y, test_input_x)
    writeCsvFile("shuyuel2_7.csv", test_output_y)

    # 8. 10 TREES + 16 DEPTH + STRETCHED
    test_output_y = randomForest(10, 16, train_input_x_stretched, train_input_y, test_input_x_stretched)
    writeCsvFile("shuyuel2_8.csv", test_output_y)

    # 9. 30 TREES + 4 DEPTH + UNTOUCHED
    test_output_y = randomForest(30, 4, train_input_x, train_input_y, test_input_x)
    writeCsvFile("shuyuel2_9.csv", test_output_y)

    # 10. 30 TREES + 4 DEPTH + STRETCHED
    test_output_y = randomForest(30, 4, train_input_x_stretched, train_input_y, test_input_x_stretched)
    writeCsvFile("shuyuel2_10.csv", test_output_y)

    # # 11. 30 TREES + 16 DEPTH + UNTOUCHED
    test_output_y = randomForest(30, 16, train_input_x, train_input_y, test_input_x)
    writeCsvFile("shuyuel2_11.csv", test_output_y)

    # 12. 30 TREES + 16 DEPTH + STRETCHED
    test_output_y = randomForest(30, 16, train_input_x_stretched, train_input_y, test_input_x_stretched)
    writeCsvFile("shuyuel2_12.csv", test_output_y)










