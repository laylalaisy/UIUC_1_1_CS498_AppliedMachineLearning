import csv
import numpy as np

def readInData(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        list = []
        for row in reader:
            list.append(row)
        file.close()

        array = np.array(list)
        return array

if __name__ == '__main__':
    ## READ IN DATA
    dataI_in = readInData("./Data/dataI.csv")
    # dataII_in = readInData("./Data/dataII.csv")
    # dataIII_in = readInData("./Data/dataIII.csv")
    # dataIV_in = readInData("./Data/dataIV.csv")
    # dataV_in = readInData("./Data/dataV.csv")
    # iris_in = readInData("./Data/iris.csv")

