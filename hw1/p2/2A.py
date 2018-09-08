import csv
import numpy as np

if __name__ == "__main__":

    # read in input_data
    with open("./train.csv", "r") as input_file:
        input_reader = csv.reader(input_file)

        input_data_list = []
        for row in input_reader:
            input_data_list.append(row)

        input_file.close()

    input_data = np.array(input_data_list)

    input_y = input_data[1:, 1:2]

    print(input_y)
