import random
from scipy.io import loadmat
import scipy
import pandas as pd
import numpy as np
from numpy.random import randn
import pyeeg
import pandas as pd

import TFConvNetwork


mat = loadmat('Data/EEG_DATA/A01T.mat')  # load mat-fil
variables = mat['data']

runs_data = []
runs_class = []

NUMBER_OF_CYCLES_ON_EACH = 312


for run in range(3, len(variables[0])):
    runs_data.append(variables[0][run][0][0][0])

for run in range(len(runs_data)):
    print("Run " + str(run))
    run_class = []
    temp = np.array(variables[0][run+3][0][0][1])
    currindex = 0
    count = 0
    for time in range(len(runs_data[run])):
        if time in temp:
            count = 0
            itemindex = np.where(temp == time)
            run_class.append(variables[0][run+3][0][0][2][itemindex][0])
            currindex = variables[0][run+3][0][0][2][itemindex][0]
        else:
            count += 1
            if count >= NUMBER_OF_CYCLES_ON_EACH:
                currindex = 0

            run_class.append(currindex)

    runs_class.append(run_class)

print("Data combined")

## Create true data points 
data = []
labels = []
blankLabel = np.zeros(5)

unique, counts = np.unique(runs_class[0], return_counts=True)

leastVal = 100 #int(round(counts.min()/3,1)) #counts.min()
print(leastVal)

for i in range(len(unique)):
    print("Class " + str(i))
    locations = np.where(runs_class[0] == unique[i])[0]
    locations = locations[np.where((locations > 19) & (locations < (len(runs_class[0])-19)))[0]][0:leastVal]
    np.random.shuffle(locations)

    temp = np.array([np.copy(blankLabel)])
    temp[0][i] = 1

    if (i == 0):
        totalclass = np.repeat(temp, leastVal, axis=0)
        totaldata =  runs_data[0][locations[0]-19:locations[0]+20]
        locations = locations[1:]
    else:
        totalclass = np.append(totalclass, np.repeat(temp, leastVal, axis=0), axis=0)

    for j in locations:
        totaldata = np.dstack((totaldata, runs_data[0][j-19:j+20]))

print("Time windows stacked")

dataArray = np.array(totaldata)
labelsArray = np.array(totalclass)

rng_state = np.random.get_state()
np.random.shuffle(dataArray)
np.random.set_state(rng_state)
np.random.shuffle(labelsArray)

print("Data points shuffled")


    

'''
totalclass = []
for i in range(len(unique)):
    totalCount = 0
    iterate = 0
    while totalCount <= leastVal:
        while (runs_class[0][iterate] != i):
            iterate+=1
        
        totaldata.append(runs_data[0][iterate]) 
        totalclass.append(runs_class[0][iterate])
        count+=1

rng_state = np.random.get_state()
np.random.shuffle(totaldata)
np.random.set_state(rng_state)
np.random.shuffle(totalclass)
    
'''

'''
vals = []
for j in limitedData:
    vals.append(runs_data[0][j])

print(vals)
'''
'''
dataIndex = random.randint(19, len(runs_data[0])-19)
dataChunk = np.array(runs_data[0][dataIndex-19:dataIndex+20])
data.append(dataChunk)
temp = np.copy(blankLabel)
temp[runs_class[0][dataIndex]] = 1
labels.append(temp)
'''

del mat
del variables

splitPoint = int(round((np.max(dataArray.shape)/5)*4,1))
print("Train/Test split point: " + str(splitPoint))
CNN = TFConvNetwork.TFConvNetwork(len(runs_data[0][0]), 39, 5, dataArray[:,:,0:splitPoint], labelsArray[0:splitPoint], dataArray[:,:,splitPoint:np.max(dataArray.shape)], labelsArray[splitPoint:np.max(dataArray.shape)])
CNN.trainAndClassify(1000)
'''
At this point we have a data and labels array
The data array has the EEG data stored per run
The labels array has one of 5 labels -- 0 if no event
                                     -- 1-4 for other events

    
'''