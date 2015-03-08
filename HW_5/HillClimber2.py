# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 09:35:37 2015

@author: amodr_000
"""

import random
import math

#import numpy as np

step = 0.0000001

numsteps = 1000

def rnum():
    return random.uniform(-1.0, 1.0)

randnum = rnum()

def f(inarr):
    return math.exp(-((inarr[0])**2 + (inarr[1])**2))

def makearray(n, lower, upper, etype='Float'):
    arr = []
    if etype == 'Float':
        for i in range(n):
            arr.append(random.uniform(lower, upper))
    return arr
    
def grad(point, stepsize):
    arr = []
    if type(stepsize) == float:
        for i in range(len(point)):
            subarr = []
            for j in range(len(point)):
                if i == j:
                    subarr.append(point[i] + stepsize)
                else:
                    subarr.append(0)
            arr.append(subarr)
        out = []
        for i in range(len(arr)):
            psum = []
            for j in range(len(arr[i])):
                psum.append(point[j] + arr[i][j])
            out.append((f(psum) - f(point))/stepsize)
    return out
    
def cpyarr(a):
    out = []
    for i in a:
        out.append(i)
    return out
        
random.seed(1970)

randarray = []
oldarray = []
newarray = []

oldarray = makearray(2, -1.0, 1.0)

print oldarray

newarray = cpyarr(oldarray)

for thinking in range(numsteps):
    oldarray = cpyarr(newarray)
    
    newarray = []
    
    g = grad(oldarray, step)
    
    gstep = []    
    for i in range(len(g)):
        gstep.append(g[i] * step)    

    for i in range(len(oldarray)):
        newarray.append(oldarray[i] - gstep[i])
    
    print f(newarray), f(oldarray), gstep
    
    if f(newarray) < f(oldarray):
        print 'shrink step'
        newarray = cpyarr(oldarray)
        step = step / 10

print newarray, 'newarray'