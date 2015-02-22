# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 13:06:30 2015

@author: team8
"""

import csv
import numpy as np

classes = 10
features = 256

def colrowexp(v1):
    """Expands vector into single matrix. Not normalized"""
    v1 = np.matrix(v1)
    v1t = np.matrix.transpose(v1)

    vout = v1t * v1
    return vout

arrtrain = []
arrtest = []

#Open the training data set. Turned into a csv by Libre Office
with open('zip.train.csv', 'rb') as csvfile:
    f = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in f:
        arrtrain.append(row)

#Open the test data set. Turned into a csv by Libre Office        
with open('zip.test.csv', 'rb') as csvfile:
    f = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in f:
        arrtest.append(row)

#Load training array. 
training = []
for i in arrtrain:
    temp = []
    for j in i:
        temp.append(float(j))
    training.append((temp[1:], int(temp[0])))

#Load test array    
testing = []
for i in arrtest:
    temp = []
    for j in i:
        temp.append(float(j))
    testing.append((temp[1:], int(temp[0])))

#Create zeroes array for mu for all classes   
mu = []
for i in range(classes):
    mu.append([0.0]*features)

#Create zeroes array for storing the count for each set of training data
count = [0]*classes

#Form mu for each class    
for i in training:
    count[i[1]] += 1
    mu[i[1]] = np.add(mu[i[1]], i[0])

#Normalize mu    
for i in range(len(mu)):
    mu[i] = mu[i]/float(count[i])

#Set up the array of sigma matrices with zeroes
fred = training[0][0]
sig = colrowexp(fred)
sigma1 = np.zeros_like(sig)
sigma = []
for i in range(classes):
    sigma.append(sigma1)

#Create the classes sigma arrays    
for i in training:
    ivec = i[1]
    x_i = i[0]
    mu_i = mu[ivec]
    currvec = np.subtract(x_i, mu_i)
    sigma[ivec] = np.add(sigma[ivec], colrowexp(currvec))

#Normalize each sigma    
for i in range(len(sigma)):
    sigma[i] = sigma[i]/float(count[i])
    
print sigma[0]

print mu[0]

np.save("sigmazip", sigma)
np.save("muzip", mu)

np.save("trainingzip", training)
np.save("testingzip", testing)