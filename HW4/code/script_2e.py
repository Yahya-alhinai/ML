import numpy as np
from statistics import mean, mode
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math

Train = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/optdigits_train.txt', 'r').read().split()
Test = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/optdigits_test.txt', 'r').read().split()

TrainD = list()
TrainR = list()

TestD = list()
TestR = list()

for i in range(len(Train)):
    Train[i] = np.array(Train[i].split(","), dtype=float)
    TrainD.append(Train[i][0:64])
    TrainR.append(Train[i][64])
    
for i in range(len(Test)):
    Test[i] = np.array(Test[i].split(","), dtype=float)
    TestD.append(Test[i][0:64])
    TestR.append(Test[i][64])


TrainD_mean = list()
for i in range(len(np.transpose(TrainD))):
    TrainD_mean.append(mean(np.transpose(TrainD)[i]))

TrainD_mean = np.transpose(TrainD_mean)

s = np.cov(np.transpose(TrainD-np.transpose(TrainD_mean)))

eigenvalues, components = LA.eig(s)


eigenvalues = np.sort(eigenvalues, axis=None)[::-1] 
# eigenvalues = eigenvalues/sum(eigenvalues)

eigV = list()
for i in components:
   eigV.append(i[0:9]) 

Ztrain = np.transpose(np.transpose(eigV) @  np.transpose(TrainD))
Ztest = np.transpose(np.transpose(eigV) @  np.transpose(TestD))


colors  = ['red','sandybrown','navy','c','purple','pink','Aqua', 'green','blue','orange']

for i in range(len(Ztrain)):
    plt.scatter(Ztrain[i,0], -Ztrain[i,1], TrainR[i], color=[colors[int(TrainR[i])]])
    
for i in range(int(len(Ztrain)/15)):
    plt.text(Ztrain[i,0], -Ztrain[i,1], str(int(TrainR[i])))
plt.show() 


for i in range(len(Ztest)):
    plt.scatter(Ztest[i,0], -Ztest[i,1], TrainR[i], color=[colors[int(TrainR[i])]])
    
for i in range(int(len(Ztest)/3)):
    plt.text(Ztest[i,0], -Ztest[i,1], str(int(TestR[i])))
plt.show() 
