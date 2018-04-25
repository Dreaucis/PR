import numpy as np
from numpy import random


class WrappedGaussianDist:
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
    def prob(self,x,nSummands = 5):
        # WrappedNormalPdf: Gives an approximation of the wrapped normal distribution via 5 summands.

        x = np.asmatrix(x)
        pdf_sum = 0
        for i in range(0,5):
            x_i = x + 2*np.pi*(i-np.floor(nSummands/2))
            pdf_i = np.transpose((1 / (np.sqrt(2 * np.pi * np.square(self.stDev))))) *\
                    np.exp( -np.square(x_i - self.mean) / (2 * np.square(self.stDev)))
            pdf_sum = pdf_sum + pdf_i

        return pdf_sum

def testWrappedGaussianDist():
    wgd = WrappedGaussianDist(np.pi, 1)

    a = [np.pi,0,-np.pi,4]
    b = np.asmatrix(a)
    c = np.concatenate((b,b))
