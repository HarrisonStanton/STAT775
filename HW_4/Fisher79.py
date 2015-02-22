# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:39:09 2015

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
    inter = 1.0/(2 * math.pi * sigma)
    expon = (-0.5 * (x - mu) ** 2)/(sigma ** 2)
    return inter * math.exp(expon)

#Array of (0, 0) points so we can sum outputs
conf_arr = np.zeros((10, 10), int)

ident = np.identity(256)

shifter = ident * np.finfo(float).eps

sigmasub = np.matrix(la.inv(sigma[9] - sigma[7] - shifter))

musub = np.matrix(mu[9] - mu[7])

a = sigmasub * np.transpose(musub)

a = np.divide(a, la.norm(a))

train7 = []
train9 = []

for i in training:
    if i[1] == 7:
        train7.append(i[0])
    if i[1] == 9:
        train9.append(i[0])

aprime = []

for i in a:
    aprime.append(float(i))        
        
mean7 = np.dot(mu[7], aprime)

sd7 = math.sqrt(float(np.transpose(a) * sigma[7] * a))

mean9 = np.dot(mu[9], aprime)
    
sd9 = math.sqrt(float(np.transpose(a) * sigma[9] * a))

count7 = 0
count9 = 0

for i in range(len(train7)):
    testing = np.dot(train7[i], aprime)
    is7 = probofx(testing, mean7, sd7)
    
    is9 = probofx(testing, mean9, sd9)

    if is7 > is9:
        count7 += 1
    else:
        count9 += 1

print count7, count9, float(count7)/float(count7+count9)

for i in range(len(train9)):
    testing = np.dot(train9[i], aprime)
    is7 = probofx(testing, mean7, sd7)
    
    is9 = probofx(testing, mean9, sd9)

    if is7 > is9:
        count7 += 1
    else:
        count9 += 1
        
print count7, count9, float(count9)/float(count7+count9)