import numpy as np
from statistics import mean, mode
from numpy import linalg as LA
import matplotlib.pyplot as plt


Train = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/optdigits_train.txt', 'r').read().split()
Test = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/optdigits_test.txt', 'r').read().split()

TN = list()
TT = list()
for i in range(len(Train)):
    Train[i] = np.array(Train[i].split(","), dtype=float)
    TN.append(Train[i][0:64])
    
for i in range(len(Test)):
    Test[i] = np.array(Test[i].split(","), dtype=float)
    TT.append(Test[i][0:64])

s = np.cov(np.transpose(TN))
eigenvalues, components = LA.eig(s)


eigenvalues = np.sort(eigenvalues, axis=None)[::-1] 
eigenvalues = eigenvalues/sum(eigenvalues)


inc = [0.0] * 64
inc[0] = eigenvalues[0]
for i in range(1, len(eigenvalues)):
    inc[i] = eigenvalues[i] + inc[i-1]
    

plt.plot(inc)
plt.xlabel('Eigenvector')
plt.ylabel('Proportion of Variance')


MinK = 0;
for i in range(len(inc)):
    if (inc[i] >= 0.9):
        MinK = i + 1;
        break


print("Minimum number of eigenvectors that explain at least 90% of the variance = {}".format(MinK))


TN_mean = list()
for i in range(len(np.transpose(TN))):
    TN_mean.append(mean(np.transpose(TN)[i]))

TT_mean = list()
for i in range(len(np.transpose(TT))):
    TT_mean.append(mean(np.transpose(TT)[i]))

components_k = list()
for i in range(len(components)):
    components_k.append(components[i][0:21])

z_train = (np.transpose(np.transpose(components_k) @ np.transpose((TN - np.transpose(TN_mean))))).tolist()
z_test = (np.transpose(np.transpose(components_k) @ np.transpose((TT - np.transpose(TT_mean))))).tolist()


for i in range(len(z_train)):
    z_train[i].append(Train[i][64])



for i in range(len(z_test)):
    z_test[i].append(Test[i][64])


z_train = np.array(z_train)
z_test = np.array(z_test)


for k in range(1,8,2):
    
    dist = 0.0
    idx = [[0 for i in range(k)] for j in range(len(z_test))]
    
    minVal = [[float("inf") for i in range(k)] for j in range(len(z_test))]

    for i in range(len(z_test)):
        for j in range(len(z_train)):
            dist = np.linalg.norm(z_train[j][0:21] - z_test[i][0:21])
            
            for l in range(k):
                if dist < minVal[i][l]:
                    for q in reversed(range(l+1,k)):
                        idx[i][q] = idx[i][q-1]
                    idx[i][l] = z_train[j][21]
                    minVal[i][l] = dist
                    break
            
    count = 0
    for i in range(len(z_test)):
        if mode(idx[i]) != z_test[i][21]:
            count+=1

    E = count/len(z_test)

    print("k = {} - Error Rate: {}".format(k, E))
    
plt.show() 
