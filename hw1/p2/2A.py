import csv
import numpy as np

if __name__ == "__main__":

    # read in training csv file
    with open("./train.csv", "r") as input_file:
        input_reader = csv.reader(input_file)
        # read in training data
        input_data_list = []
        for row in input_reader:
            input_data_list.append(row)
        # close training csv file
        input_file.close()

    # change input data from list to array
    input_data = np.array(input_data_list)

    # extract data of feature and label
    input_x = input_data[1:, 2:]
    input_y = input_data[1:, 1:2]

    print(input_x.shape)
    print(input_y.shape)
