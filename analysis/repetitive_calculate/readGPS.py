import numpy as np

import matplotlib.pyplot as plt

def getDists(Xs,Ys):
    dists = [0]
    previousPoint = None
    initialized = 0
    for (x,y) in zip(Xs,Ys):
        #print (x,y)
        if initialized == 0:
            previousPoint = (x,y)
            initialized = 1
        else:
            dist = ((x-previousPoint[0])**2 + (y-previousPoint[1])**2)**0.5
            dists.append(dist)
            previousPoint = (x,y)
    return dists



def getBx(gps_fp,downsample=1):
    f = open(gps_fp)
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
    x = np.array(x)
    y = np.array(y)
    x = x - x[0]
    y = y - y[0]
    x = x[::downsample]
    y = y[::downsample]
    #print x
    #print y
    dists = getDists(x,y)
    return dists

if __name__=="__main__":
    path = '/home/leo/Downloads/2019_dataset/row1_meta/'
    gps_fp = path + 'vehicle_pose.txt'
    f = open(gps_fp)
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        x.append(float(line.split(',')[0]))
        y.append(float(line.split(',')[1]))
    x = np.array(x)
    y = np.array(y)
    x = x - x[0]
    y = y - y[0]
    #print x
    #print y
    dists = getDists(x,y)
    print np.mean(dists)
    print np.max(dists)
    print np.min(dists)
    
    plt.plot(x,y,'o')
    plt.show()


