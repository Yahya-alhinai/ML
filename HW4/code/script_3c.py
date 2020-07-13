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


Data = list()
Data.extend(TrainD)
Data.extend(TestD)


Data_mean = list()
for i in range(len(np.transpose(Data))):
    Data_mean.append(mean(np.transpose(Data)[i]))

Data_mean = np.transpose(Data_mean)

s = np.cov(np.transpose(Data-np.transpose(Data_mean)))

eigenvalues, components = LA.eig(s)

eigenvalues = np.sort(eigenvalues, axis=None)[::-1] 

fig=plt.figure(figsize=(12,8))
count = 1

for k in [10, 50, 100]:
    zt = np.real(np.transpose(np.transpose(components[:,0:k]) @ np.transpose(Data[0:5]-Data_mean)));
    xhat = np.real(np.transpose(components[:,0:k] @ np.transpose(zt)));

    for i in range(5):
        fig.add_subplot(3, 5, count)
        plt.title('Image {} - K = {}'.format(i+1,k))
        plt.imshow(np.real(np.reshape(xhat[i], (30, 32))))
        count = count+1
        
plt.show()

