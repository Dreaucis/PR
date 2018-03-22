from numpy.random import multivariate_normal
import numpy as np
from numpy import random


class GaussianDist:
    def __init__(self,mean = 0, stDev = 1,dataSize=1):
        self.mean = mean
        self.stDev = stDev
        self.dataSize = dataSize
    def getMean(self):
        return self.mean
    def getStDev(self):
        return self.stDev
    def getDataSize(self):
        return self.dataSize
    def setMean(self,mean):
        self.mean = mean
    def setStDev(self,stDev):
        self.stDev = stDev
    def setDataSize(self,dataSize):
        self.dataSize = dataSize
    def generate(self):
        random.normal(self.mean,self.stDev,self.dataSize)
    def prob(self,x):
        return (1/(np.sqrt(2*np.pi*self.stDev**2)))*np.exp(-(x-self.mean)**2/(2*self.stDev**2))

def logProb(pD,x):
    nObj = len(pD)
    nX = 1
    if not isinstance(x,int):
        nX = len(x)
    logP = np.zeros((nObj,nX))
    for i in range(0,nObj):
        dSize = 1
        if not isinstance(pD[i].getMean(),int):
            dSize = len(pD[i].getMean())
        if dSize == 1:
            z = (x-np.repeat(pD[i].getMean(),nX))
            z = np.divide(z,np.repeat(pD[i].getStDev(),nX))
            logP[i,:] = -np.sum(np.multiply(z,z),0)/2
            logP[i,:] = logP[i,:] -np.sum(np.log(pD[i].getStDev())) -dSize*np.log(2*np.pi)/2
        else:
            print('ERROR')
    return logP
#TODO: Fix so that it can take a vecotr of distributions as input!