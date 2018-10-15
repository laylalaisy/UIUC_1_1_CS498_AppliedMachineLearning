import os
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC


act_num = 14

def readData(path1):
    # activities: data of all activities
    # activity: data of each activity
    # cur_file: data of each file
    # cur_act: data of each line
    folders = os.listdir(path1)
    folders = [x for x in folders if 'MODEL' not in x and 'DS_Store' not in x]

    print(folders)

    activities = []
    for i in range(act_num):
        activity = []
        path2 = "" + path1 + "/" + folders[i]
        files = os.listdir(path2)
        random_select = random.sample(range(len(files)), len(files))
        for j in random_select:
            file = files[j]
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
            segment_num = math.floor(length/segment_size)         # number of segment

            for k in range(segment_num):                # build up segment
                train_activity.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:segment_size*3+1])
                activities.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:segment_size*3+1])
            train_activity = np.array(train_activity)
            train_activities.append(train_activity)

        for j in range(train_num, train_num+test_num):                      # traverse each train file
            test_activity = []
            cur_file = act_data[i][j]                   # current file's data
            length = cur_file.shape[0]                  # number of samples in current file
            segment_num = math.floor(length/segment_size)         # number of segment

            for k in range(segment_num):                # build up segment
                test_activity.append(cur_file[k*segment_size:(k+1)*segment_size].T.flatten()[:segment_size*3+1])
            test_activity = np.array(test_activity)
            test_activities.append(test_activity)
    activities = np.array(activities)
    train_activities = np.array(train_activities)
    test_activities = np.array(test_activities)

    return activities, train_activities, test_activities

def createTrainHistogram(data, cluster_size, labels, segment_size):
    length = data.shape[0]
    index = 0

    histograms = []
    signals = []
    for i in range(length):
        signal = data[i][0, segment_size*3]
        signals.append(signal)
        count = np.zeros(cluster_size)
        for j in range(data[i].shape[0]):
            label = labels[index]
            count[label] = count[label] + 1
            index += 1
        histograms.append(count)
    histograms = np.array(histograms)
    signals = np.array(signals)

    for i in range(length):
        total = np.sum(histograms[i])
        for j in range(cluster_size):
            histograms[i][j] = float(histograms[i][j]) # normalization

    return histograms, signals


def createTestHistogram(data, cluster_size, labels):
    count = np.zeros(cluster_size)
    length = data.shape[0]

    for i in range(length):
        label = labels[i]
        count[label] = count[label] + 1

    total = np.sum(count)
    for i in range(cluster_size):
        count[i] = float(count[i])

    return count

def kms(activities, cluster_size, segment_size):
    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(activities[:, :segment_size*3])
    train_centers = kmeans.cluster_centers_
    train_labels = kmeans.labels_
    return kmeans, train_labels, train_centers

def kmsPredict(model, data):
    return model.predict(data)

def getCenter(data ,labels, cluster_size, segment_size):
    centers = np.array([np.zeros(segment_size*3)]*cluster_size)
    label_count = defaultdict(float)
    for i in range(data.shape[0]):
        label, point = labels[i], data[i]
        centers[label] += point
        label_count[label] += 1
    for i in range(cluster_size):
        centers[i] = centers[i] / label_count[i]
    return centers

def getLable(data, centers):
    dist, label = float('inf'), 0
    for i in range(centers.shape[0]):
        cur_dist = np.linalg.norm(data-centers[i])
        if cur_dist < dist:
            dist, label = cur_dist ,i 
    return np.array([label])

def agg(activities, cluster_size, segment_size):
    cluster = AgglomerativeClustering(n_clusters=cluster_size, affinity='euclidean', linkage='ward')
    train_labels = cluster.fit_predict(activities[:, :segment_size*3])
    train_centers = getCenter(activities[:, :segment_size*3], train_labels, cluster_size, segment_size)
    return train_labels, train_centers

def draw(histograms, signals, cluster_size, act_name):
    for i in range(14):
        histogram_sum, count = np.zeros(cluster_size), 0.0
        for (histogram, signal) in zip(histograms, signals):
            if signal == i:
                histogram_sum += histogram
                count += 1.0
        histogram_sum /= count
        plt.bar(range(cluster_size), histogram_sum)
        plt.title('Histogram of ' + act_name[i])
        plt.savefig('%s.png'%i)
        plt.close()
        
def execute(act_name, act_data, segment_size, cluster_size, percent, matrix_output):
    activities, act_train, act_test = splitData(act_data, percent, segment_size)

    model, train_labels, train_centers = kms(activities, cluster_size, segment_size)
    # train_labels, train_centers = agg(activities, cluster_size, segment_size)

    train_histogram, train_signal = createTrainHistogram(act_train, cluster_size, train_labels, segment_size)
    draw(train_histogram, train_signal, cluster_size, act_name)

    test_labels = []
    test_samples = act_test.shape[0]
    for i in range(test_samples):                       # each file
        test_label = []
        for j in range(act_test[i].shape[0]):           # each segment
            test_label.append(kmsPredict(model, act_test[i][j, :segment_size*3].reshape(1, -1)))
            # test_label.append(getLable((act_test[i][j, :segment_size*3]), train_centers))
        test_label = np.array(test_label)
        test_labels.append(test_label)
    test_labels = np.array(test_labels)

    test_histogram = []
    for i in range(test_samples):
        test_histogram.append(createTestHistogram(act_test[i], cluster_size, test_labels[i]))

    # rf = RF(max_depth=5, random_state=0).fit(train_histogram, train_signal)
    svm = SVC(gamma='auto').fit(train_histogram, train_signal)

    accurate = 0
    cov_matrix = dict()
    for i in range(test_samples):
        # label = rf.predict(test_histogram[i].reshape(1, -1))[0]
        label = svm.predict(test_histogram[i].reshape(1, -1))[0]
        label_ori = act_test[i][0, segment_size*3]
        # print(type(label),label,type(label_ori))
        if label_ori not in cov_matrix:
            cov_matrix[label_ori] = [0]*14
        cov_matrix[label_ori][label] += 1
        if int(label) == label_ori:
            accurate = accurate + 1
    cov_df = pd.DataFrame.from_dict(cov_matrix, orient='index', 
                                    columns=[str(x) for x in range(14)])
    cov_df.to_csv('cov.csv',index=False)
    print(segment_size, cluster_size, percent, 'kmeans', accurate/ len(act_test))
    print(cov_df, cov_df.sum())


if __name__ == "__main__":
    act_name, act_data = readData('./HMP_Dataset')

    execute(act_name, act_data, segment_size=32, cluster_size=24, percent=0.8, matrix_output=True)
