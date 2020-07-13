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

s = np.cov(np.transpose(Data))
eigenvalues, components = LA.eig(s)


eigenvalues = np.sort(eigenvalues, axis=None)[::-1] 
eigenvalues = eigenvalues/sum(eigenvalues)


inc = [0.0] * len(eigenvalues)
inc[0] = eigenvalues[0]
for i in range(1, len(eigenvalues)):
    inc[i] = eigenvalues[i] + inc[i-1]
    
plt.plot(np.real(inc))
plt.xlabel('Eigenvector')
plt.ylabel('Proportion of Variance')

MinK = 0;
for i in range(len(inc)):
    if (inc[i] >= 0.9):
        MinK = i + 1;
        break


print("Minimum number of eigenvectors that explain at least 90% of the variance = {}".format(MinK))


TrainD_mean = list()
for i in range(len(np.transpose(TrainD))):
    TrainD_mean.append(mean(np.transpose(TrainD)[i]))

TestD_mean = list()
for i in range(len(np.transpose(TestD))):
    TestD_mean.append(mean(np.transpose(TestD)[i]))

components_k = list()
for i in range(len(components)):
    components_k.append(components[i][0:MinK])

z_train = (np.transpose(np.transpose(components_k) @ np.transpose((TrainD - np.transpose(TrainD_mean))))).tolist()
z_test = (np.transpose(np.transpose(components_k) @ np.transpose((TestD - np.transpose(TestD_mean))))).tolist()

for i in range(len(z_train)):
    z_train[i].append(Train[i][960])



for i in range(len(z_test)):
    z_test[i].append(Test[i][960])


z_train = np.array(z_train)
z_test = np.array(z_test)

Mink = 41
for k in range(1,8,2):
    
    dist = 0.0
    idx = [[0 for i in range(k)] for j in range(len(z_test))]
    
    minVal = [[float("inf") for i in range(k)] for j in range(len(z_test))]

    for i in range(len(z_test)):
        for j in range(len(z_train)):
            dist = np.linalg.norm(z_train[j][0:Mink] - z_test[i][0:Mink])
            
            for l in range(k):
                if dist < minVal[i][l]:
                    for q in reversed(range(l+1,k)):
                        idx[i][q] = idx[i][q-1]
                    idx[i][l] = z_train[j][Mink]
                    minVal[i][l] = dist
                    break
            
    count = 0
    for i in range(len(z_test)):
        if mode(idx[i]) != z_test[i][Mink]:
            count+=1

    E = count/len(z_test)

    print("k = {} - Error Rate: {}".format(k, E))
    
plt.show() 




