import os
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as RF


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

def splitData(act_data, percent, segment_size):
    activities = []
    train_activities = []
    test_activities = []

    for i in range(act_num): # each activity
        file_num = act_data[i].shape[0]
        test_num = math.floor(file_num*(1-percent))                 # number of test files/signals
        if(test_num < 1):
            test_num = 1
        train_num = file_num-test_num                   # number of train files/signals

        for j in range(train_num):                      # traverse each train file
            train_activity = []
            cur_file = act_data[i][j]                   # current file's data
            length = cur_file.shape[0]                  # number of samples in current file
            segment_num = math.floor(length/32)         # number of segment

            for k in range(segment_num):                # build up segment
                train_activity.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:97])
                activities.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:97])
            train_activity = np.array(train_activity)
            train_activities.append(train_activity)

        for j in range(train_num, train_num+test_num):                      # traverse each train file
            test_activity = []
            cur_file = act_data[i][j]                   # current file's data
            length = cur_file.shape[0]                  # number of samples in current file
            segment_num = math.floor(length/32)         # number of segment

            for k in range(segment_num):                # build up segment
                test_activity.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:97])
            test_activity = np.array(test_activity)
            test_activities.append(test_activity)
    activities = np.array(activities)
    train_activities = np.array(train_activities)
    test_activities = np.array(test_activities)

    return activities, train_activities, test_activities

def createTrainHistogram(data, cluster_size, centers, labels):
    length = data.shape[0]
    index = 0


    histograms = []
    signals = []
    for i in range(length):
        signal = data[i][0, 96]
        signals.append(signal)
        count = np.zeros(cluster_size)
        for j in range(data[i].shape[0]):
            label = labels[index]
            count[label] = count[label] + 1
        histograms.append(count)
    histograms = np.array(histograms)
    signals = np.array(signals)

    for i in range(length):
        total = np.sum(histograms[i])
        for j in range(cluster_size):
            histograms[i][j] = float(histograms[i][j] / total)

    return histograms, signals


def createTestHistogram(data, cluster_size, centers, labels):
    count = np.zeros(cluster_size)
    length = data.shape[0]

    for i in range(length):
        label = labels[i]
        count[label] = count[label] + 1

    total = np.sum(count)
    for i in range(cluster_size):
        count[i] = float(count[i] / total)

    return count


def execute(act_data, segment_size, cluster_size, percent, matrix_output):
    activities, act_train, act_test = splitData(act_data, percent, segment_size)

    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(activities[:, :96])
    train_centers = kmeans.cluster_centers_
    train_labels = kmeans.labels_

    train_histogram, train_signal = createTrainHistogram(act_train, cluster_size, train_centers, train_labels)

    test_labels = []
    test_samples = act_test.shape[0]
    for i in range(test_samples):                       # each file
        test_label = []
        for j in range(act_test[i].shape[0]):           # each segment
            test_label.append(kmeans.predict(act_test[i][j, :96].reshape(1, -1)))
        test_label = np.array(test_label)
        test_labels.append(test_label)
    test_labels = np.array(test_labels)

    test_histogram = []
    for i in range(test_samples):
        test_histogram.append(createTestHistogram(act_test[i], cluster_size, train_centers, test_labels[i]))

    rf = RF(max_depth=5, random_state=0).fit(train_histogram, train_signal)

    accurate = 0
    for i in range(test_samples):
        label = rf.predict(test_histogram[i].reshape(1, -1))
        label_ori = act_test[i][0, 96]
        if int(label) == label_ori:
            accurate = accurate + 1
    print(label, label_ori)


if __name__ == "__main__":
    act_name, act_data = readData('./HMP_Dataset')

    execute(act_data, segment_size=32, cluster_size=400, percent=0.9, matrix_output=True)





