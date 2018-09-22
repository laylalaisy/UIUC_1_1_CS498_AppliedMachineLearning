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

def getMean(data_in):
    return np.mean(data_in, axis=0)

def reconstruct(data_in, data_mean):
    # NORMALIZATION
    data_mean_repeat = np.tile(data_mean, [samples, 1])
    data_norm = (data_in - data_mean_repeat)

    # COVARIANCE
    data_cov = np.cov(data_norm, rowvar=0)

    # EIGENVALUE, EIGENVECTOR
    data_eigval, data_eigvec = np.linalg.eig(data_cov)

    # REDUCE DIMENSION AND RECONSTRUCTION
    data_reconstructions = []
    for dim in range(dims + 1):
        data_feature = data_eigvec[:, :dim].reshape(dims, dim)
        data_reduce = np.dot(data_norm, data_feature).reshape(samples, dim)
        data_reconstruction = np.dot(data_reduce, data_feature.T) + data_mean_repeat
        data_reconstructions.append(data_reconstruction)

    return data_reconstructions

def MSE(x1, x2):
    mse = []
    for dim in range(dims+1):
        square = pow((x1[dim] - x2), 2)
        sum = np.sum(square, axis=1)
        mean = round(np.mean(sum), 3)
        mse.append(mean)
    return mse

if __name__ == '__main__':
    mse = []

    # iris
    iris_in = readInData('./Data/iris.csv')
    iris_mean = getMean(iris_in)

    # dataI
    mse_I = []
    dataI_in = readInData('./Data/dataI.csv')
    dataI_mean = getMean(dataI_in)
    # dataI + n
    dataI_n_reconstructions = reconstruct(dataI_in, iris_mean)
    dataI_n_mse = MSE(dataI_n_reconstructions, iris_in)
    mse_I.append(dataI_n_mse)
    # dataI + c
    dataI_c_reconstructions = reconstruct(dataI_in, dataI_mean)
    dataI_c_mse = MSE(dataI_c_reconstructions, iris_in)
    mse_I.append(dataI_c_mse)
    # mse
    mse.append(mse_I)

    # dataII
    mse_II = []
    dataII_in = readInData('./Data/dataII.csv')
    dataII_mean = getMean(dataII_in)
    # dataII + n
    dataII_n_reconstructions = reconstruct(dataII_in, iris_mean)
    dataII_n_mse = MSE(dataII_n_reconstructions, iris_in)
    mse_II.append(dataII_n_mse)
    # dataII + c
    dataII_c_reconstructions = reconstruct(dataII_in, dataII_mean)
    dataII_c_mse = MSE(dataII_c_reconstructions, iris_in)
    mse_II.append(dataII_c_mse)
    # mse
    mse.append(mse_II)

    # dataIII
    mse_III = []
    dataIII_in = readInData('./Data/dataIII.csv')
    dataIII_mean = getMean(dataIII_in)
    # dataIII + n
    dataIII_n_reconstructions = reconstruct(dataIII_in, iris_mean)
    dataIII_n_mse = MSE(dataIII_n_reconstructions, iris_in)
    mse_III.append(dataIII_n_mse)
    # dataIII + c
    dataIII_c_reconstructions = reconstruct(dataIII_in, dataIII_mean)
    dataIII_c_mse = MSE(dataIII_c_reconstructions, iris_in)
    mse_III.append(dataIII_c_mse)
    # mse
    mse.append(mse_III)

    # dataIV
    mse_IV = []
    dataIV_in = readInData('./Data/dataIV.csv')
    dataIV_mean = getMean(dataIV_in)
    # dataIV + n
    dataIV_n_reconstructions = reconstruct(dataIV_in, iris_mean)
    dataIV_n_mse = MSE(dataIV_n_reconstructions, iris_in)
    mse_IV.append(dataIV_n_mse)
    # dataIV + c
    dataIV_c_reconstructions = reconstruct(dataIV_in, dataI_mean)
    dataIV_c_mse = MSE(dataIV_c_reconstructions, iris_in)
    mse_IV.append(dataIV_c_mse)
    # mse
    mse.append(mse_IV)

    # dataV
    mse_V = []
    dataV_in = readInData('./Data/dataV.csv')
    dataV_mean = getMean(dataV_in)
    # dataV + n
    dataV_n_reconstructions = reconstruct(dataV_in, iris_mean)
    dataV_n_mse = MSE(dataV_n_reconstructions, iris_in)
    mse_V.append(dataV_n_mse)
    # dataV + c
    dataV_c_reconstructions = reconstruct(dataV_in, dataI_mean)
    dataV_c_mse = MSE(dataV_c_reconstructions, iris_in)
    mse_V.append(dataV_c_mse)
    # mse
    mse.append(mse_V)

    # WRITE dataII
    with open("shuyuel2-recon.csv", "w") as output_file:
        output_writer = csv.writer(output_file)
        # write header
        fileHeader = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
        output_writer.writerow(fileHeader)
        # write content
        output_writer.writerows(dataII_c_reconstructions[2])


    # WRITE MSE
    with open("shuyuel2-numbers.csv", "w") as output_file:
        output_writer = csv.writer(output_file)
        # write header
        fileHeader = ["0N", "1N", "2N", "3N", "4N", "0c", "1c", "2c", "3c", "4c"]
        output_writer.writerow(fileHeader)
        # write content
        content = []
        for dim in range(dims+1):
            temp = []
            for iter in range(dims+1):
                temp.append(mse[dim][0][iter])
            for iter in range(dims+1):
                temp.append(mse[dim][1][iter])
            content.append(temp)
        output_writer.writerows(content)





