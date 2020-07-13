import numpy as np
from statistics import mean 

for k in [1, 2, 3]:
    if k == 1:
        T1 = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/training_data1.txt', 'r').read().split()
        Test1 = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/test_data1.txt', 'r').read().split()
    elif k == 2:
        T1 = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/training_data2.txt', 'r').read().split()
        Test1 = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/test_data2.txt', 'r').read().split()
    elif k == 3:
        T1 = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/training_data3.txt', 'r').read().split()
        Test1 = open('G:/My Drive/SCHOOL/School/CSCI 5521/HW2/resources/test_data3.txt', 'r').read().split()


    T1C1 = list()
    T1C2 = list()
    for i in range(len(Test1)):
        Test1[i] = np.array(Test1[i].split(","), dtype=float)

    for i in range(len(T1)):
        T1[i] = np.array(T1[i].split(","), dtype=float)
        if T1[i][8] == 1:
            T1C1.append(T1[i][0:8])
        else:
            T1C2.append(T1[i][0:8])
            

    PC1 = len(T1C1)/len(T1)
    PC2 = len(T1C2)/len(T1)

    s = [0] * 8
    M1 = [0] * 8
    for i in T1C1:
        for j in range(0, 8):
            s[j] += i[j]
    for j in range(0, 8):
            M1[j] = s[j]/len(T1C1)

    s = [0] * 8
    M2 = [0] * 8
    for i in T1C2:
        for j in range(0, 8):
            s[j] += i[j]
    for j in range(0, 8):
            M2[j] = s[j]/len(T1C2)
            
            
    s = [0] * 8
    M = [0] * 8
    for i in T1:
        for j in range(0, 8):
            s[j] += i[j]
    for j in range(0, 8):
            M[j] = s[j]/len(T1)

    #||--------------- M1 -------------------||
    S1 = np.cov(np.transpose(T1C1))
    S2 = np.cov(np.transpose(T1C2))
    S1M1 = S1
    S2M1 = S2

    g1 = [0] * len(Test1)
    g2 = [0] * len(Test1)
    g = [0] * len(Test1)

    for i in range(len(Test1)):
        g1[i] = np.log(PC1)-(8/2)*np.log(2*np.pi)-(1/2)*np.log(np.linalg.det(S1))-(1/2) * (Test1[i][0:8]-M1) @ np.linalg.inv(S1) @ np.transpose(Test1[i][0:8]-M1)
        g2[i] = np.log(PC2)-(8/2)*np.log(2*np.pi)-(1/2)*np.log(np.linalg.det(S2))-(1/2) * (Test1[i][0:8]-M2) @ np.linalg.inv(S2) @ np.transpose(Test1[i][0:8]-M2)
        g[i] = np.log(g1[i]/g2[i])

    for i in range(len(Test1)):
        if g1[i] > g2[i]:
            g[i] = 1
        else:
            g[i] = 2

    count = 0
    for i in range(len(Test1)):
        if g[i] != Test1[i][8]:
            count+=1
    
    E1 = count/len(Test1)


    #||--------------- M2 -------------------||
    S = (S1+S2)/2
    S1M2 = S2M2 = S
    for i in range(len(Test1)):
        g1[i] = np.log(PC1)-(1/2) * (Test1[i][0:8]-M1) @ np.linalg.inv(S) @ np.transpose(Test1[i][0:8]-M1)
        g2[i] = np.log(PC2)-(1/2) * (Test1[i][0:8]-M2) @ np.linalg.inv(S) @ np.transpose(Test1[i][0:8]-M2)
        g[i] = np.log(g1[i]/g2[i])
        
    for i in range(len(Test1)):
        if g1[i] > g2[i]:
            g[i] = 1
        else:
            g[i] = 2

    count = 0
    for i in range(len(Test1)):
        if g[i] != Test1[i][8]:
            count+=1
                        
    E2 = count/len(Test1)


    #||--------------- M3 -------------------||


    a1 = 0
    for i in range(len(T1C1)):
        a1 += ((T1C1[i]-M1) @ np.transpose(T1C1[i]-M1) / (8*len(T1C1)))

    a2 = 0
    for i in range(len(T1C2)):
        a2 += ((T1C2[i]-M2) @ np.transpose(T1C2[i]-M2) / (8*len(T1C2)))

    S1 = np.eye(8)*a1
    S2 = np.eye(8)*a2

    for i in range(len(Test1)):
        g1[i] = np.log(PC1)-(8/2)*np.log(2*np.pi)-(1/2)*8*np.log(a1) - (1/2*a1) * (Test1[i][0:8]-M1) @ np.transpose(Test1[i][0:8]-M1)
        g2[i] = np.log(PC2)-(8/2)*np.log(2*np.pi)-(1/2)*8*np.log(a2) - (1/2*a2) * (Test1[i][0:8]-M2) @ np.transpose(Test1[i][0:8]-M2)


    for i in range(len(g)):
        if np.log(g1[i]/g2[i]) > 0:
            g[i] = 1
        else:
            g[i] = 2

    count = 0
    for i in range(len(Test1)):
        if g[i] != Test1[i][8]:
            count+=1

    E3 = count/len(Test1)


    #||--------------- results -------------------||
    dash = '-' * 25
    dashL = '-' * 60

    print(dashL)
    print(dashL)
    print("Data Set {}".format(k))
    print(dashL)
    print("P(C1) = {}".format(PC1))
    print("P(C2) = {}".format(PC2))
    print("M1 = {:>5.4f}, {:>5.4f}, {:>5.4f}, {:>5.4f}, {:>5.4f}, {:>5.4f}, {:>5.4f}, {:>5.4f}".format(M1[0],M1[1],M1[2],M1[3],M1[4],M1[5],M1[6],M1[7]))
    print("M2 = {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}".format(M2[0],M2[1],M2[2],M2[3],M2[4],M2[5],M2[6],M2[7]))

    print(dash)
    print("Data Set {} - Model 1".format(k))
    print(dash)
    print("Error Rate = {:>5.4f}  ".format(E1))
    print("S1 = ")
    for i in S1M1:
        print("{:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}".format(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]))
    print("\n")

    print("S2 = ")
    for i in S2M1:
        print("{:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}".format(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]))

    print(dash)
    print("Data Set {} - Model 2".format(k))
    print(dash)
    print("Error Rate = {:>5.4f}  ".format(E2))

    print("S1 = ")
    for i in S1M2:
        print("{:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}".format(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]))
    print("\n")

    print("S2 = ")
    for i in S2M2:
        print("{:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}  {:>5.4f}".format(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]))
        
    print(dash)
    print("Data Set {} - Model 3".format(k))
    print(dash)
    print("Error Rate = {:>5.4f}  ".format(E3))

    print("σ1 = {}".format(a1))
    print("σ2 = {}".format(a2))
    print("\n")
