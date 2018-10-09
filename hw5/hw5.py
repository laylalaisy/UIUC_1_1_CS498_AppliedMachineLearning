import os


if __name__ == "__main__":
    path1 = './HMP_Dataset'
    folders = os.listdir(path1)

    activities = []
    for i in range(len(folders)):
        path2 = ""+path1+"/"+folders[i]
        files = os.listdir(path2)
        for file in files:
            if not os.path.isdir(file):
                with open(""+path2+"/"+file, 'r') as f:
                    for line in f.readlines():
                        activity = []
                        num = str(line).rstrip("\r\n").split(" ")
                        activity.append(int(num[0]))
                        activity.append(int(num[1]))
                        activity.append(int(num[2]))
                        activity.append(i)
                        activities.append(activity)

    print(activities)




