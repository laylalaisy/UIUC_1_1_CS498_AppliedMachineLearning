import os
import numpy as np
import scipy as sp
from sklearn import cluster
import matplotlib.pyplot as plt

def readData(path1):
    folders = os.listdir(path1)

    activities = []
    for i in range(len(folders)):
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

    print(activities[0])



if __name__ == "__main__":
    readData('./HMP_Dataset')





