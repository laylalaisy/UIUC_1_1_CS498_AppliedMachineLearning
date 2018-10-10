import os
import numpy as np
import scipy as sp
from sklearn import cluster
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

act_num = 14

def readData(path1):
    folders = os.listdir(path1)

    activities = []
    for i in range(act_num):
        activity = []
        path2 = "" + path1 + "/" + folders[i]
        files = os.listdir(path2)
        for file in files:
            if not os.path.isdir(file):
                with open("" + path2 + "/" + file, 'r') as f:
                    for line in f.readlines():
                        current = []
                        num = str(line).rstrip("\r\n").split(" ")
                        current.append(int(num[0]))
                        current.append(int(num[1]))
                        current.append(int(num[2]))
                        current.append(i)
                        current = np.array(current)
                        activity.append(current)
        activity = np.array(activity)
        activities.append(activity)
    activities = np.array(activities)

    return folders, activities

def split(act_data, percent, segment_size):
    for i in range(act_num):
        length = act_data[i].shape[0]
        print(length)



def execute(act_data, segment_size=32, cluster_size=40*12, percent=0.9, matrix_output=True):
    #act_train, act_test =
    split(act_data, percent, segment_size)


if __name__ == "__main__":
    act_name, act_data = readData('./HMP_Dataset')

    execute(act_data, segment_size=32, cluster_size=40*12, percent=0.9, matrix_output=True)





