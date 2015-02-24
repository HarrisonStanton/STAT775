#23, 45, 79

import numpy as np

import matplotlib.pyplot as plt
import numpy.linalg as la

import math

sigma = np.load("sigmazip.npy")
mu = np.load("muzip.npy")

training = np.load("trainingzip.npy")
testing = np.load("testingzip.npy")

#Check any pairs you like.
Numbers = [(2, 3), (4, 5), (7, 9)]

in_array = testing

distancearray = []

tracking = []

#Change this value to test the pairs in Numbers. 0 index
for problem in range(3):

    first = Numbers[problem][0]
    second = Numbers[problem][1]
    
    points = [first, second]
    
    def probofx(x, mu, sigma):
        inter = 1.0/(math.sqrt(2 * math.pi * sigma))
        expon = (-0.5 * (x - mu) ** 2)/(sigma ** 2)
        return inter * math.exp(expon)
    
    def buildu(sigma, mu, first, second):
        """Builds a based on sigma, mu, first and second values under 
        investigation"""
        #Prebuild arrays and catch singular matrix and try epsilon I matrix
        try:
            sigmasub = np.matrix(la.inv(sigma[second] + sigma[first]))
        except np.linalg.LinAlgError:
            sigmasub = np.matrix(la.inv(sigma[second] + sigma[first] + 
                np.identity(len(sigma[second]))*np.finfo(float).eps))
        
        musub = np.matrix(mu[second] - mu[first])
        
        #Put them together and then normalize to unit vector. a \equiv u
        a = sigmasub * np.transpose(musub)    
        a = np.divide(a, la.norm(a))
        
        return a
        
    a = buildu(sigma, mu, first, second)
    
    #Python was being a butt so I had to go through this little song and 
    #dance to get floats in the array, not an array of array points. Don't ask.
    aprime = []
    for i in a:
        aprime.append(float(i))
    
    train = []
    test = []
    
    mean = [[], []]
    sd = [[], []]
    count = [0, 0]
    
    #Build empty array
    for i in range(10):
        train.append([])    
    #Add training data to matrix it belongs in
    for i in training:
        train[i[1]].append(i[0])        
    
    #This was getting silly since note they use different mu.        
    mean[0] = np.dot(mu[first], aprime)
    sd[0] = math.sqrt(float(np.transpose(a) * sigma[first] * a))
    
    mean[1] = np.dot(mu[second], aprime)
    sd[1] = math.sqrt(float(np.transpose(a) * sigma[second] * a))
    
    print "Training Data"
    #And this is it
    for order in points:
        count = [0, 0]
        for i in range(len(train[order])):
            
            ifis = [0, 0]
            whichvalue = np.dot(train[order][i], aprime)
            ifis[0] = probofx(whichvalue, mean[0], sd[0])
            
            ifis[1] = probofx(whichvalue, mean[1], sd[1])
        
            if ifis[0] > ifis[1]:
                count[0] += 1
            else:
                count[1] += 1
                
        if order == first:
            print str(first) + 's', str(second) + 's', '% Accuracy'
            print count[0], count[1], float(count[0])/float(count[0]+count[1])
        else:
            print str(first) + 's', str(second) + 's', '% Accuracy'
            print count[0], count[1], float(count[1])/float(count[0]+count[1])