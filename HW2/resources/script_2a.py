import numpy as np
from statistics import mean, mode

Train = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/optdigits_train.txt', 'r').read().split()
Test = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/optdigits_test.txt', 'r').read().split()

for i in range(len(Train)):
    Train[i] = np.array(Train[i].split(","), dtype=float)
    
for i in range(len(Test)):
    Test[i] = np.array(Test[i].split(","), dtype=float)

for k in [1, 3, 5, 7]:
    dist = 0.0
    idx = [[0 for i in range(k)] for j in range(len(Test))]

    minVal = [[float("inf") for i in range(k)] for j in range(len(Test))]

    for i in range(len(Test)):
        for j in range(len(Train)):
            dist = np.linalg.norm(Train[j][0:64] - Test[i][0:64])
            
            for l in range(k):
                if dist < minVal[i][l]:
                    for q in reversed(range(l+1,k)):
                        idx[i][q] = idx[i][q-1]
                    idx[i][l] = Train[j][64]
                    minVal[i][l] = dist
                    break
            
            
    count = 0
    for i in range(len(Test)):
        if mode(idx[i]) != Test[i][64]:
            count+=1


    E = count/len(Test)

    print("k = {} - Error Rate: {}".format(k, E))
