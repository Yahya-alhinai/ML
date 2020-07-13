import numpy as np
from statistics import mean, mode
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math

Train = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/face_train_data_960.txt', 'r').read().splitlines()
Test = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/face_test_data_960.txt', 'r').read().splitlines()

TrainD = list()
TrainR = list()

TestD = list()
TestR = list()

for i in range(len(Train)):
    Train[i] = np.array(Train[i].split(" "), dtype=float)
    TrainD.append(Train[i][0:960])
    TrainR.append(Train[i][960])
    
for i in range(len(Test)):
    Test[i] = np.array(Test[i].split(" "), dtype=float)
    TestD.append(Test[i][0:960])
    TestR.append(Test[i][960])

TrainD.extend(TestD)

TrainD_mean = list()
for i in range(len(np.transpose(TrainD))):
    TrainD_mean.append(mean(np.transpose(TrainD)[i]))

TrainD_mean = np.transpose(TrainD_mean)

s = np.cov(np.transpose(TrainD-np.transpose(TrainD_mean)))

eigenvalues, components = LA.eig(s)

eigenvalues = np.sort(eigenvalues, axis=None)[::-1] 
# eigenvalues = eigenvalues/sum(eigenvalues)

fig=plt.figure(figsize=(12,3))
for i in range(5):
    fig.add_subplot(1, 5, i+1)
    plt.title('EigenFace {}'.format(1+i))
    plt.imshow(np.real(np.reshape(components[:, i], (30, 32))))

plt.show()



