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
        pX = np.zeros((self.nStates,np.shape(x)[1]))
        for i in range(0,self.nStates):
            pX[i,:] = self.emissionDist[i].prob(x)

        pX,_ = logScale(pX)
        _,c = forward(self.mc,pX)

        betaHat = backward(self.mc,pX,c)

        logP = np.sum(np.log(c))
        return logP

def MakeLeftRightHMM(nStates,obsData,lData = []):
    """
    :param nStates:
    :param pD:
    :param obsData:
    :param lData: vector with lengths of training sub-sequences.
    :return:
    """
    if len(lData) == 0:
        lData = [np.size(obsData,1)]
    pD = []
    for i in range(0, nStates):
        pD.append(GaussianDist())
    D = np.mean(lData)
    D = D/nStates
    mc = FiniteMarkovChain(); initLeftRight(mc,nStates,D)
    hmm = HiddenMarkovChain(mc,pD); initLeftRightHMM(hmm,obsData,lData)

    HMMTrain(hmm,obsData,lData)

    return hmm

def HMMTrain(hmm,xT,lxT,nIterations=5,minStep=np.log(1.01)):
    ixt = [0]+ (np.cumsum(lxT)-1).tolist()

    logprobs = np.asmatrix(np.zeros((1,nIterations)))
    logPold = 99999999999
    # TODO: continue from here
    for nTraining in range(0,nIterations):
        aState = AState(hmm)
        for r in range(0,len(lxT)):
            logP = aState.accumulateHMM(hmm, xT[:, ixt[r]:(ixt[r+1])])
        logprobs[0,nTraining] = logprobs[0,nTraining] + logP
        logPdelta = logprobs[0,nTraining] - logPold
        logPold = logprobs[0,nTraining]
        aState.adaptSetHMM(hmm)
    if not nTraining:
        nTraining = 0
    while logPdelta > minStep:
        nTraining = nTraining+1
        logprobs = np.concatenate([logprobs,np.asmatrix(0)],1)
        #logprobs[0,nTraining] = 0 TODO: REMEMBER HERE.
        aState = AState(hmm)
        for r in range(0, len(lxT)):
            logP = aState.accumulateHMM(hmm, xT[:, ixt[r]:(ixt[r + 1])])
        logprobs[0,nTraining] = logprobs[0,nTraining] + logP
        logPdelta = logprobs[0,nTraining] - logPold
        logPold = logprobs[0,nTraining]
        aState.adaptSetHMM(hmm)

def initLeftRightHMM(hmm,obsData,lData): #TODO: COMPARE WITH MATLAB FROM HERE. ERROR ON DIST
    dSize = np.size(obsData,0)
    if isinstance(lData,int):
        lData = [lData]
    nTrainingSeq = len(lData)
    startIndex = np.cumsum([1]+lData)-1
    nStates = hmm.mc.nStates()
    pD = hmm.getEmissionDist()
    for i in range(0,nStates):
        xT = np.asmatrix(np.zeros((dSize,0)))
        for r in range(0,nTrainingSeq):
            dStart = startIndex[r] + np.round((i)*lData[r]/nStates).astype(int)
            dEnd = startIndex[r] + np.round((i+1)*lData[r]/nStates).astype(int)
            xT = np.concatenate((xT,obsData[:,dStart:dEnd]),1)
        pD[i] = initDist(pD[i],xT)
    hmm.setEmissionDist(pD)

def initDist(dist,xT):
    dist.setMean(xT.mean(1))
    dist.setStDev(xT.std(1))
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
####################################### ACCUMULATE FUNCTIONS ###########################################
class AState:
    def __init__(self,hmm):
        # For MC
        self.pI = np.asmatrix(np.zeros((np.shape(hmm.mc.getInitProb()))))
        self.pS = np.asmatrix(np.zeros((np.shape(hmm.mc.getTransProb()))))
        # For Dist
        nObj = len(hmm.emissionDist)
        self.pD = list()
        for i in range(0,nObj):
            dSize = len(hmm.emissionDist[i].mean)
            temp_dict = {'sumDev':np.zeros((dSize,1)),'sumSqDev':np.zeros((dSize,dSize)),'sumWeight':0}
            self.pD.append(temp_dict)
        # logProb
        self.logProb = 0

    def accumulateMC(self,mc,pX):
        T = np.size(pX,1)
        nStates = mc.nStates()

        A = mc.getTransProb()

        alfaHat,c = forward(mc,pX)
        betaHat = backward(mc,pX,c)

        gamma = np.multiply(np.multiply(alfaHat,betaHat),np.matlib.repmat(c[0,0:T],nStates,1))

        self.pI = self.pI + gamma[:,0]

        pXbH = np.multiply(pX[:,1:],betaHat[:,1:])

        aHpXbH = alfaHat[:,0:-1]*pXbH.transpose()
        xi = np.multiply(aHpXbH,A[:,0:nStates])

        self.pS[:,0:nStates] = self.pS[:,0:nStates] + xi

        self.pS[:,nStates] = self.pS[:,nStates] + np.multiply(alfaHat[:,T-1],betaHat[:,T-1])*c[0,T-1]

        lP = np.sum(np.log(c))

        return [gamma,lP]

    def accumulateDist(self,pD,obsData,obsWeight):
        [dSize,nData] = np.shape(obsData)
        nObj = len(pD)
        for i in range(0,nObj):
            Dev = obsData - pD[i].mean
            wDev = np.multiply(Dev,obsWeight[i,:]) #TODO: might be issue, should maybe be repmat
            self.pD[i]['sumDev'] = self.pD[i]['sumDev'] + np.sum(wDev,1)
            self.pD[i]['sumSqDev'] = self.pD[i]['sumSqDev'] + Dev*np.transpose(wDev)
            self.pD[i]['sumWeight'] = self.pD[i]['sumWeight'] + np.sum(obsWeight[i,:])

    def accumulateHMM(self, hmm, obsData):
        pX = np.zeros((hmm.nStates, np.size(obsData,1)))
        for i in range(0, hmm.nStates):
            pX[i, :] = hmm.emissionDist[i].prob(obsData)
        pX = np.asmatrix(pX)
        gamma, logP = self.accumulateMC(hmm.mc,pX)
        self.accumulateDist(hmm.emissionDist,obsData,gamma)
        self.logProb = self.logProb + logP
        logP = self.logProb
        return logP

    def adaptSetHMM(self,hmm):
        self.adaptSetMC(hmm.mc)
        self.adaptSetDist(hmm.emissionDist)

    def adaptSetMC(self,mc):
        # Setting initial probability
        mc.setInitProb(self.pI/np.sum(self.pI))
        # Setting transition probability
        mc.setTransProb(np.divide(self.pS, np.repeat(np.sum(self.pS, 1), np.size(self.pS, 1),1)))

    def adaptSetDist(self,pD):
        for i in range(0,len(pD)):
            if self.pD[i]['sumWeight'] > 0:
                pD[i].mean = pD[i].mean + self.pD[i]['sumDev']/self.pD[i]['sumWeight']

                S2 = self.pD[i]['sumSqDev'] - np.divide((self.pD[i]['sumDev']*
                                                         np.transpose(self.pD[i]['sumDev'])), self.pD[i]['sumWeight'])
                covEstim = np.divide(S2, self.pD[i]['sumWeight'])
                if any(np.diag(covEstim) < 10^-10):
                    print('WARNING: ZERO DIV IN ADAPT SET DIST')
                    covEstim = np.diag(np.repeat(np.inf,np.shape(pD[i].mean)))

                pD[i].stDev = np.asmatrix(np.sqrt(np.diag(covEstim))).transpose()

        


def logScale(probMat):
    np.asmatrix(probMat)
    maxInCols = np.max(probMat,0)
    logS = np.log(maxInCols)
    scaledProbMat = np.divide(probMat,np.exp(logS))
    return scaledProbMat,logS
###############################################TEST####################################################

def test():
    initProb = np.matrix('1;0')
    transProb = np.matrix('0.9,0.1,0;0,0.9,0.1')
    x = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
    fmc = FiniteMarkovChain(initProb, transProb)
    gD1 = GaussianDist(0,1)
    gD2 = GaussianDist(3,2)
    hmm = HiddenMarkovChain(fmc,[gD1,gD2])
    a = np.exp(hmm.logProb([-0.2,2.6,1.3]))

def testInitLeftRight():
    mc = FiniteMarkovChain()
    initLeftRight(mc,5)

def testMakeLeftRightHMM():
    initProb = np.matrix('1;0')
    transProb = np.matrix('0.9,0.1,0;0,0.9,0.1')
    x = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0,0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
    fmc = FiniteMarkovChain(initProb, transProb)
    gD1 = GaussianDist(0,1)
    gD2 = GaussianDist(0,1)
    nStates = 2
    pD = [gD1,gD2]
    obsData = np.matrix('-0.2,2.6,1.3,-0.1,2.5,1.4,-0.3,2.7,1.2;-0.2,2.6,1.3,-0.1,2.5,1.4,-0.3,2.7,1.2') #TODO: Allow for multiple features
    lData = [3,3,3]
    hmm = MakeLeftRightHMM(nStates,pD,obsData,lData)#(fmc,[gD1,gD2])
    print(hmm)
    print(hmm.mc.initProb)
    print(hmm.mc.transProb)
    print(hmm.getEmissionDist()[0].mean)
    print(hmm.getEmissionDist()[0].stDev)
    print(hmm.getEmissionDist()[1].mean)
    print(hmm.getEmissionDist()[1].stDev)

