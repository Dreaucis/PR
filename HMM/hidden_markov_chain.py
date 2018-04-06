import numpy as np
import numpy.matlib
from distributions.GaussD.gaussian_dist import GaussianDist,logProb
from MC.markov_chain import forward, FiniteMarkovChain, backward


class HiddenMarkovChain:
    def __init__(self,mc,emissionDist):
        self.mc = mc
        self.emissionDist = emissionDist
        self.nStates = mc.nStates()
    def setMC(self,mc):
        self.mc = mc
    def setEmissionDist(self,emissionDist):
        self.emissionDist = emissionDist
    def getEmissionDist(self):
        return self.emissionDist
    def logProb(self,x):
        pX = np.zeros((self.nStates,len(x)))
        for i in range(0,self.nStates):
            pX[i,:] = self.emissionDist[i].prob(x)

        pX,_ = logScale(pX)
        _,c = forward(self.mc,pX)
        print(c)
        betaHat = backward(self.mc,pX,c)
        print(betaHat)
        logP = sum(np.log(c))
        return logP

def MakeLeftRightHMM(nStates,pD,obsData,lData = None):
    """
    :param nStates:
    :param pD:
    :param obsData:
    :param lData: vector with lengths of training sub-sequences.
    :return:
    """
    if lData == None:
        lData = np.size(obsData,1)
    D = np.mean(lData)
    D = D/nStates
    mc = FiniteMarkovChain(); initLeftRight(mc,nStates,D)
    hmm = HiddenMarkovChain(mc,pD); initLeftRightHMM(hmm,obsData,lData)
    return hmm


def initLeftRightHMM(hmm,obsData,lData):
    #dSize = np.size(obsData,0)
    if isinstance(lData,int):
        lData = [lData]
    nTrainingSeq = len(lData)
    startIndex = np.cumsum([1]+lData)-1
    nStates = hmm.mc.nStates()
    pD = hmm.getEmissionDist()
    print(pD)
    for i in range(0,nStates):
        xT = list()
        for r in range(0,nTrainingSeq):
            dStart = startIndex[r] + np.round((i)*lData[r]/nStates).astype(int)
            dEnd = startIndex[r] + np.round((i+1)*lData[r]/nStates).astype(int)
            xT.append(obsData[:,dStart:dEnd])
        pD[i] = initDist(pD[i],xT)
    hmm.setEmissionDist(pD)

def initDist(dist,xT):
    dist.setMean(xT[0].mean(1))
    dist.setStDev(xT[0].std(1))
    return dist

def initLeftRight(mc,nStates,stateDuration = 10):
    minDiagProb = 0.1
    D = np.max([1,stateDuration])
    aii = np.max([minDiagProb,(D-1)/D])
    aij = 1-aii
    A = np.zeros((nStates,nStates+1))
    for i in range(0,nStates):
        A[i,i] = aii
        A[i,i+1] = aij
    pO = np.zeros((nStates,1)); pO[0] = 1
    mc.setInitProb(pO)
    mc.setTransProb(A)

def testInitLeftRight():
    mc = FiniteMarkovChain()
    initLeftRight(mc,5)
    print(mc.getInitProb())

def testMakeLeftRightHMM():
    initProb = np.matrix('1;0')
    transProb = np.matrix('0.9,0.1,0;0,0.9,0.1')
    x = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
    fmc = FiniteMarkovChain(initProb, transProb)
    gD1 = GaussianDist(0,1)
    gD2 = GaussianDist(3,2)
    nStates = 2
    pD = [gD1,gD2]
    obsData = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
    hmm = MakeLeftRightHMM(nStates,pD,obsData,lData = None)#(fmc,[gD1,gD2])
    print(hmm)
    print(hmm.mc.initProb)
    print(hmm.mc.transProb)
    print(hmm.getEmissionDist()[0].mean)
    print(hmm.getEmissionDist()[0].stDev)
    print(hmm.getEmissionDist()[1].mean)
    print(hmm.getEmissionDist()[1].stDev)
testMakeLeftRightHMM()

class AState:
    def __init__(self,mc):
        self.pI = np.zeros((np.shape(mc.getInitProb())))
        self.pS = np.zeros((np.shape(mc.getTransProb())))
    def accumulate(self,mv,pX):
        T = np.size(pX,1)
        nStates = mc.nStates()
        A = mc.getTransProb()
        alfaHat,c = forward(mc,pX)
        betaHat = backward(mc,pX,c)
        gamma = np.multiply(np.multiply(alfaHat,betaHat),np.matlib.repmat(c[0,1:T],nStates,1))
        print(gamma)

        #TODO: Continue from here

def logScale(probMat):
    np.asmatrix(probMat)
    maxInCols = np.max(probMat,0)
    logS = np.log(maxInCols)
    scaledProbMat = np.divide(probMat,np.exp(logS))
    return scaledProbMat,logS

def test():
    initProb = np.matrix('1;0')
    transProb = np.matrix('0.9,0.1,0;0,0.9,0.1')
    x = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
    fmc = FiniteMarkovChain(initProb, transProb)
    gD1 = GaussianDist(0,1)
    gD2 = GaussianDist(3,2)
    hmm = HiddenMarkovChain(fmc,[gD1,gD2])
    a = np.exp(hmm.logProb([-0.2,2.6,1.3]))

