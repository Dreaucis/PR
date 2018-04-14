import numpy as np
class FiniteMarkovChain():
    def __init__(self,initProb = None,transProb = None):
        self.initProb = np.asmatrix(initProb)
        self.transProb = np.asmatrix(transProb)
        self.isFinite = True

    def nStates(self):
        return np.size(self.transProb,0)
    def setInitProb(self,initProb):
        self.initProb = np.asmatrix(initProb)
    def setTransProb(self,transProb):
        self.transProb = np.asmatrix(transProb)
    def getInitProb(self):
        return self.initProb
    def getTransProb(self):
        return self.transProb

def forward(fmc,pX):
    """
    :param fmc: FiniteMarkovChain
    :param pX: numpy matriix with entries  pX(j,t) = P(x_t|s_t = j). 
    Probability of observation given state.
    :return: alfahat, c
    """
    T = np.size(pX,1)
    A = fmc.getTransProb()
    q = fmc.getInitProb()
    alfaTemp = np.asmatrix(np.zeros(np.shape(pX)))
    pX = np.matrix(pX)
    alfaTemp[:,0] = np.multiply(q,pX[:,0])
    c = np.zeros((1,T+1))
    AExit = np.asmatrix(A[:,-1])
    A = A[:,0:-1]
    c[0,0] = np.sum(alfaTemp[:,0],0)
    alfaHat = np.asmatrix(np.zeros(np.shape(pX)))
    alfaHat[:,0] = alfaTemp[:,0]/c[0,0]

    if T > 1:
        for t in range(1,T):
            alfaTemp[:,t] = np.multiply(pX[:,t],(alfaHat[:,t-1].transpose()*A).transpose())
            c[0,t] = np.sum(alfaTemp[:,t])
            alfaHat[:,t] = alfaTemp[:,t]/c[0,t]

        c[0,T] = alfaHat[:,T-1].transpose()*AExit

    return alfaHat,c

def backward(fmc,pX,c):
    T = np.size(pX, 1)
    nS = fmc.nStates()
    A = fmc.getTransProb()
    betaHat = np.asmatrix(np.zeros((nS,T)))
    for i in range(0,nS):
        betaHat[i,T-1] = A[i,nS]/(c[0,T-1]*c[0,T])
    for t in range(T-2,-1,-1):
        for i in range(0,nS):
            summ = 0
            for j in range(0,nS):
                summ = summ + A[i,j]*pX[j,t+1]*betaHat[j,t+1]
            betaHat[i,t] = summ/c[0,t]
    return betaHat

def test():
    initProb = np.matrix('1;0;0')
    transProb = np.matrix('0.5,0.5,0,0;0,0.5,0.5,0;0,0,0.5,0.5')
    pX = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0;0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2;0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
    fmc = FiniteMarkovChain(initProb, transProb)
    alfahat,c = forward(fmc,pX)

