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
   eigV.append(i[0:2]) 

Ztrain = np.transpose(np.transpose(eigV) @  np.transpose(TrainD))
Ztest = np.transpose(np.transpose(eigV) @  np.transpose(TestD))


colors  = ['red','sandybrown','navy','c','purple','pink','Aqua', 'green','blue','orange']

for i in range(len(Ztrain)):
    plt.scatter(Ztrain[i,0], -Ztrain[i,1], TrainR[i], color=[colors[int(TrainR[i])]])
    
for i in range(len(Test)):
    plt.text(Ztrain[i,0], -Ztrain[i,1], str(int(TrainR[i])))
plt.show() 


for i in range(len(Ztest)):
    plt.scatter(Ztest[i,0], -Ztest[i,1], TrainR[i], color=[colors[int(TrainR[i])]])
    
for i in range(len(Ztest)):
    plt.text(Ztest[i,0], -Ztest[i,1], str(int(TestR[i])))
plt.show() 



# inc = [0.0] * 64
# inc[0] = eigenvalues[0]
# for i in range(1, len(eigenvalues)):
#     inc[i] = eigenvalues[i] + inc[i-1]
    

# plt.plot(inc)
# plt.xlabel('Eigenvector')
# plt.ylabel('Proportion of Variance')


# MinK = 0;
# for i in range(len(inc)):
#     if (inc[i] >= 0.9):
#         MinK = i + 1;
#         break


# print("Minimum number of eigenvectors that explain at least 90% of the variance = {}".format(MinK))


# TN_mean = list()
# for i in range(len(np.transpose(TN))):
#     TN_mean.append(mean(np.transpose(TN)[i]))

# TT_mean = list()
# for i in range(len(np.transpose(TT))):
#     TT_mean.append(mean(np.transpose(TT)[i]))

# components_k = list()
# for i in range(len(components)):
#     components_k.append(components[i][0:21])

# z_train = (np.transpose(np.transpose(components_k) @ np.transpose((TN - np.transpose(TN_mean))))).tolist()
# z_test = (np.transpose(np.transpose(components_k) @ np.transpose((TT - np.transpose(TT_mean))))).tolist()


# for i in range(len(z_train)):
#     z_train[i].append(Train[i][64])



# for i in range(len(z_test)):
#     z_test[i].append(Test[i][64])


# z_train = np.array(z_train)
# z_test = np.array(z_test)


# for k in range(1,8,2):
    
#     dist = 0.0
#     idx = [[0 for i in range(k)] for j in range(len(z_test))]
    
#     minVal = [[float("inf") for i in range(k)] for j in range(len(z_test))]

#     for i in range(len(z_test)):
#         for j in range(len(z_train)):
#             dist = np.linalg.norm(z_train[j][0:21] - z_test[i][0:21])
            
#             for l in range(k):
#                 if dist < minVal[i][l]:
#                     idx[i][l] = z_train[j][21]
#                     minVal[i][l] = dist
#                     break
            
#     count = 0
#     for i in range(len(z_test)):
#         if mode(idx[i]) != z_test[i][21]:
#             count+=1

#     E = count/len(z_test)

#     print("k = {} - Error Rate: {}".format(k, E))
    
# plt.show() 
