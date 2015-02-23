# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:30:56 2015

@author: amodr_000
"""

#23, 45, 79

import numpy as np

import matplotlib.pyplot as plt
import numpy.linalg as la

import math

sigma = np.load("sigmazip.npy")
mu = np.load("muzip.npy")

training = np.load("trainingzip.npy")
testing = np.load("testingzip.npy")

in_array = testing

distancearray = []

tracking = []

def probofx(x, mu, sigma):
    inter = 1.0/math.sqrt((2 * math.pi * sigma))
    expon = (-0.5 * (x - mu) ** 2)/(sigma ** 2)
    return inter * math.exp(expon)

#Array of (0, 0) points so we can sum outputs
conf_arr = np.zeros((10, 10), int)

sigmasub = np.matrix(la.inv(sigma[5] + sigma[4]))

musub = np.matrix(mu[5] - mu[4])

a = sigmasub * np.transpose(musub)

a = np.divide(a, la.norm(a))

train4 = []
train5 = []

for i in training:
    if i[1] == 4:
        train4.append(i[0])
    if i[1] == 5:
        train5.append(i[0])

aprime = []

for i in a:
    aprime.append(float(i))        
        
mean4 = np.dot(mu[4], aprime)

sd4 = math.sqrt(float(np.transpose(a) * sigma[4] * a))

mean5 = np.dot(mu[5], aprime)
    
sd5 = math.sqrt(float(np.transpose(a) * sigma[5] * a))

count4 = 0
count5 = 0

for i in range(len(train4)):
    testing = np.dot(train4[i], aprime)
    is4 = probofx(testing, mean4, sd4)
    
    is5 = probofx(testing, mean5, sd5)

    if is4 > is5:
        count4 += 1
    else:
        count5 += 1

print count4, count5, float(count4)/float(count4+count5)

count4 = 0
count5 = 0

for i in range(len(train5)):
    testing = np.dot(train5[i], aprime)
    is4 = probofx(testing, mean4, sd4)
    
    is5 = probofx(testing, mean5, sd5)

    if is4 > is5:
        count4 += 1
    else:
        count5 += 1
        
print count4, count5, float(count5)/float(count4+count5)