import csv
import numpy as np

dims = 4
samples = 150

def readInData(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        list = []
        for row in reader:
            list.append(row)
        file.close()

        array = np.array(list[1:]).astype(np.float)
        return array

def getMean(x):
    return    # mean of each column

if __name__ == '__main__':


    # dataI
    # READ IN DATA
    dataI_in = readInData("./Data/dataI.csv")

    # NORMALIZATION
    dataI_mean = np.mean(dataI_in, axis=0)
    dataI_std = np.std(dataI_in, axis=0)
    dataI_mean_repeat = np.tile(dataI_mean, [samples, 1])
    dataI_std_repeat = np.tile(dataI_std, [samples, 1])
    dataI_norm = (dataI_in - dataI_mean_repeat) / dataI_std_repeat

    # COVARIANCE
    dataI_cov = np.zeros([dims, dims])
    for row in range(dims):
        for col in range(dims):
            dataI_cov[row][col] = dataI_cov[col][row] = np.cov(dataI_norm[:, row], dataI_norm[:, col])[0][1]

    # iris
    # READ IN DATA
    iris_in = readInData("./Data/iris.csv")

    # NORMALIZATION
    iris_mean = np.mean(iris_in, axis=0)
    iris_std = np.mean(iris_in, axis=0)
    iris_mean_repeat = np.tile(iris_mean, [samples, 1])
    iris_std_repeat = np.tile(iris_std, [samples, 1])
    iris_norm = (iris_in - iris_mean_repeat) / iris_std_repeat

    # COVARIANCE
    iris_cov = np.zeros([dims, dims])
    for row in range(dims):
        for col in range(dims):
            iris_cov[row][col] = iris_cov[col][row] = np.cov(iris_norm[:, row], iris_norm[:, col])[0][1]



