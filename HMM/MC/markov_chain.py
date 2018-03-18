import numpy as np
class FiniteMarkovChain():
    def __init__(self,initProb,transProb):
        self.initProb = initProb
        self.transProb = transProb
    def nStates(self):
        return np.size(self.transProb,1)
    def setInitProb(self,initProb):
        self.initProb = initProb
    def setTransProb(self,transProb):
        self.transProb = transProb
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
    alfaTemp[:,0] = np.multiply(q,pX[:,0])

    c = np.zeros((1,T+1))
    AExit = A[:,-1]
    A = A[:,0:-1]

    c[0,0] = np.sum(alfaTemp[:,0],0)
    alfaHat = np.asmatrix(np.zeros(np.shape(pX)))
    alfaHat[:,0] = alfaTemp[:,0]/c[0,0]


    for t in range(1,T-1):
        alfaTemp[:,t] = np.multiply(pX[:,t],(alfaHat[:,t-1].transpose()*A).transpose())
        c[0,t] = np.sum(alfaTemp[:,t])
        alfaHat[:,t] = alfaTemp[:,t]/c[0,t]

    c[0,T] = alfaHat[:,T-1].transpose()*AExit

    return alfaHat,c

def test()
    initProb = np.matrix('1;0;0')
    transProb = np.matrix('0.5,0.5,0,0;0,0.5,0.5,0;0,0,0.5,0.5')
    pX = np.matrix('1,0.8,0.7,0.6,0.5,0.0,0.0,0.0;0.0,0.2,0.3,0.4,0.5,0.9,0.7,0.2;0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.8')
    fmc = FiniteMarkovChain(initProb,transProb)
    alfahat,c = forward(fmc,pX)
    print(alfahat)
    print(c)
