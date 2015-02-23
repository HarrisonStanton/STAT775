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
    inter = 1.0/(math.sqrt(2 * math.pi * sigma))
    expon = (-0.5 * (x - mu) ** 2)/(sigma ** 2)
    return inter * math.exp(expon)

#Array of (0, 0) points so we can sum outputs
conf_arr = np.zeros((10, 10), int)

sigmasub = np.matrix(la.inv(sigma[3] + sigma[2]))

musub = np.matrix(mu[3] - mu[2])
a = sigmasub * np.transpose(musub)

a = np.divide(a, la.norm(a))

pretrain2 = []
pretrain3 = []

train2 = [0.0] * 256
train3 = [0.0] * 256

for i in training:
    if i[1] == 2:
        pretrain2.append(i[0])
    if i[1] == 3:
        pretrain3.append(i[0])

aprime = []

for i in a:
    aprime.append(float(i))        
        
mean2 = np.dot(mu[2], aprime)

sd2 = math.sqrt(float(np.transpose(a) * sigma[2] * a))

mean3 = np.dot(mu[3], aprime)
    
sd3 = math.sqrt(float(np.transpose(a) * sigma[3] * a))

count2 = 0
count3 = 0

for i in range(len(pretrain2)):
    testing = np.dot(pretrain2[i], aprime)
    is2 = probofx(testing, mean2, sd2)
    
    is3 = probofx(testing, mean3, sd3)

    if is2 > is3:
        count2 += 1
    else:
        count3 += 1

print count2, count3, float(count2)/float(count2+count3)

count2 = 0
count3 = 0


for i in range(len(pretrain3)):
    testing = np.dot(pretrain3[i], aprime)
    is2 = probofx(testing, mean2, sd2)
    
    is3 = probofx(testing, mean3, sd3)

    if is2 > is3:
        count2 += 1
    else:
        count3 += 1
        
print count2, count3, float(count3)/float(count2+count3)