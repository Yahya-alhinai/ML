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



cdata = [[] for i in range(10)]
mi = [[] for i in range(10)]
si = [[0.0] for i in range(10)]
sw = [[0.0] * 64] * 64
sb = [[0.0] * 64] * 64


for i in range(10):
    for j in range(len(TrainR)):
        if i == int(TrainR[j]):
            cdata[i].append(TrainD[j])
            
    for j in range(64):
        mi[i].append(mean(np.transpose(cdata[i])[j]))
    ss = (np.transpose((cdata[i])-np.transpose(mi[i])) @ (((cdata[i])-np.transpose(mi[i]))))
    
    for i in range(64):
        sw[i] += ss[i]


mu = np.mean(mi, axis = 0)


for i in range(10):
    st = list()
    for q in range(10):
        st.append(np.subtract(mi[i], mi[q]))
    sb = np.add(sb, len(cdata[i]) * (np.transpose(st) @ st))

s = np.linalg.pinv(sw) @ sb 



eigenvalues_B, components = LA.eig(s)

eigenvalues = np.real(np.sort(eigenvalues_B, axis=None)[::-1] )

n = list()
for i in range(len(eigenvalues)):
    result = np.where(np.real(eigenvalues_B) == eigenvalues[i])
    n.append(result[0][0])
    


Ds = np.zeros((64, 64), float)
np.fill_diagonal(Ds, eigenvalues)

Vs = components[:,n]

comp = Vs[:, 0:9]
eigV = eigenvalues[0:9]


Ztrain = np.real(TrainD @  comp)
Ztest = np.real(TestD @  comp)

for l in [2, 4, 9]:
    Ntrain = list()
    Ntest = list()
    for i in range(len(Ztrain)):
        Ntrain.append(np.append(Ztrain[i][0:l], TrainR[i]))
        
        
    for i in range(len(Ztest)):
        Ntest.append(np.append(Ztest[i][0:l], TestR[i]))
    # Ztrain = np.concatenate(np.transpose(Ztrain), (TrainR), 0)

    exit
    Ntrain = np.array(Ntrain)
    Ntest = np.array(Ntest)

    Mink = l
    for k in [1, 3, 5]:
        
        dist = 0.0
        idx = [[0 for i in range(k)] for j in range(len(Ntest))]
        
        minVal = [[float("inf") for i in range(k)] for j in range(len(Ntest))]

        for i in range(len(Ntest)):
            for j in range(len(Ntrain)):
                dist = np.linalg.norm(Ntrain[j][0:Mink] - Ntest[i][0:Mink])
                
                for l in range(k):
                    if dist < minVal[i][l]:
                        for q in reversed(range(l+1,k)):
                            idx[i][q] = idx[i][q-1]
                        idx[i][l] = Ntrain[j][Mink]
                        minVal[i][l] = dist
                        break
                
        count = 0
        for i in range(len(Ntest)):
            if mode(idx[i]) != Ntest[i][Mink]:
                count+=1

        E = count/len(Ntest)

        print("[L = {} - k = {}] = Error Rate: {}".format(Mink, k, E))
        
    plt.show() 
