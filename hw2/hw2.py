import csv
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

    # get training dataset size
    [sample_amount, feature_amount] = train_input_data.shape

    
