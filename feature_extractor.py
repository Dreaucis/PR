import numpy as np
from numpy import linalg as LA

class FeatureExtractor:
    def __init__(self,cords):
        self.cords = cords
        self.normCords = self.normalizer()

    def normalizer(self):
        maxCords = self.cords.max(1)
        minCords = self.cords.min(1)
        diffCords = maxCords - minCords
        centCords = self.cords - (maxCords+minCords)/2

        if LA.norm(diffCords) != 0:
            if diffCords[0] == 0:
                normalizedCords = centCords
                normalizedCords[1,:] = centCords[1,:]/(maxCords[1]-minCords[1])
            elif diffCords[1] == 0:
                normalizedCords = centCords
                normalizedCords[0,:] = centCords[0,:]/(maxCords[0]-minCords[0])
            else:
                normalizedCords = centCords/(maxCords-minCords)
            return normalizedCords
        else:
            print('Singular point')
            return centCords

    def dist2cent(self):
        euclDist = LA.norm(self.normCords,2,0)
        return euclDist

a = np.matrix('1,2,3,4;,1,1,1,1')
b = FeatureExtractor(a)
print(b.normalizer())
print(b.dist2cent())
