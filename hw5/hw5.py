import os
import math
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

act_num = 14

def readData(path1):
    # activities: data of all activities
    # activity: data of each activity
    # cur_file: data of each file
    # cur_act: data of each line
    folders = os.listdir(path1)

    activities = []
    for i in range(act_num):
        activity = []
        path2 = "" + path1 + "/" + folders[i]
        files = os.listdir(path2)
        for file in files:
            cur_file = []
            if not os.path.isdir(file):
                with open("" + path2 + "/" + file, 'r') as f:
                    for line in f.readlines():
                        cur_act = []
                        num = str(line).rstrip("\r\n").split(" ")
                        cur_act.append(int(num[0]))
                        cur_act.append(int(num[1]))
                        cur_act.append(int(num[2]))
                        cur_act.append(i)
                        cur_act = np.array(cur_act)
                        cur_file.append(cur_act)
                    cur_file = np.array(cur_file)
                    activity.append(cur_file)
        activity = np.array(activity)
        activities.append(activity)

    activities = np.array(activities)
    return folders, activities

def split(act_data, percent, segment_size):
    activities = []
    train_activities = []
    test_activities = []

    for i in range(act_num): # each activity
        train_activity = []
        test_activity = []

        file_num = act_data[i].shape[0]
        test_num = math.floor(file_num*(1-percent))                 # number of test files/signals
        if(test_num < 1):
            test_num = 1
        train_num = file_num-test_num                   # number of train files/signals

        for j in range(train_num):                      # traverse each train file
            cur_file = act_data[i][j]                   # current file's data
            length = cur_file.shape[0]                  # number of samples in current file
            segment_num = math.floor(length/32)         # number of segment

            for k in range(segment_num):                # build up segment
                train_activity.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:97])

        for j in range(train_num, train_num+test_num):                      # traverse each train file
            cur_file = act_data[i][j]                   # current file's data
            length = cur_file.shape[0]                  # number of samples in current file
            segment_num = math.floor(length/32)         # number of segment

            for k in range(segment_num):                # build up segment
                test_activity.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:97])

        train_activity = np.array(train_activity)
        train_activities.append(train_activity)
        test_activity = np.array(test_activity)
        test_activities.append(test_activity)

    train_activities = np.array(train_activities)
    test_activities = np.array(test_activities)

    return train_activities, test_activities


def execute(act_data, segment_size, cluster_size, percent, matrix_output):
    act_train, act_test = split(act_data, percent, segment_size)

    print(act_train[0])

    # kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(act_train[:, :96])
    # values = kmeans.cluster_centers_
    # labels = kmeans.labels_



if __name__ == "__main__":
    act_name, act_data = readData('./HMP_Dataset')

    execute(act_data, segment_size=32, cluster_size=40*12, percent=0.9, matrix_output=True)





