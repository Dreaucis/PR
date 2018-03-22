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
        pX = np.exp(logProb(self.emissionDist,x))
        _,c = forward(self.mc,pX)
        logP = sum(np.log(c))
        return logP

initProb = np.matrix('1;0;0')
transProb = np.matrix('0.5,0.5,0,0;0,0.5,0.5,0;0,0,0.5,0.5')
x = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
fmc = FiniteMarkovChain(initProb,transProb)
gD = GaussianDist()
hmm = HiddenMarkovChain(fmc,[gD])
hmm.logProb(1)

