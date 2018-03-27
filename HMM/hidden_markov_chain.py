import numpy as np
from distributions.GaussD.gaussian_dist import GaussianDist,logProb
from MC.markov_chain import forward, FiniteMarkovChain
class HiddenMarkovChain:
    def __init__(self,mc,emissionDist):
        self.mc = mc
        self.emissionDist = emissionDist
        self.nStates = mc.nStates()
    def setMC(self,mc):
        self.mc = mc
    def setEmissionDist(self,emissionDist):
        self.emissionDist = emissionDist
    def logProb(self,x):
        pX = np.zeros((self.nStates,len(x)))
        for i in range(0,self.nStates):
            pX[i,:] = self.emissionDist[i].prob(x)

        pX,_ = logScale(pX)
        _,c = forward(self.mc,pX)
        print(c)
        logP = sum(np.log(c))
        return logP

def logScale(probMat):
    np.asmatrix(probMat)
    maxInCols = np.max(probMat,0)
    logS = np.log(maxInCols)
    scaledProbMat = np.divide(probMat,np.exp(logS))
    return scaledProbMat,logS

initProb = np.matrix('1;0')
transProb = np.matrix('0.9,0.1,0;0,0.9,0.1')
x = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
fmc = FiniteMarkovChain(initProb,transProb)
gD1 = GaussianDist(0,1)
gD2 = GaussianDist(3,2)
hmm = HiddenMarkovChain(fmc,[gD1,gD2])
a = np.exp(hmm.logProb([-0.2,2.6,1.3]))

