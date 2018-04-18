import numpy as np
from numpy import linalg as LA

class FeatureExtractor:
    def __init__(self,cords):
        self.cords = self.interpolateCords(cords, 100)
        self.normCords = self.normalizer()

    def normalizer(self):
        maxCords = self.cords.max(1)
        minCords = self.cords.min(1)
        diffCords = maxCords - minCords
        centCords = self.cords - (maxCords+minCords)/2

        if LA.norm(diffCords) != 0:
            normalizedCords = centCords/(max(maxCords-minCords))
            return normalizedCords
        else:
            print('Singular point')
            return centCords

    def interpolateCords(self,cords,nInterPoints= 100):
        # TODO: Maybe have this on all features, instead of here. ALSO IS WRONG...
        nPoints = np.size(cords,1)
        points = np.arange(nPoints)
        interPoints = np.arange(nInterPoints)*(nPoints-1)/nInterPoints
        interXCord = np.interp(interPoints, points, np.ravel(cords[0, :]))
        interYCord = np.interp(interPoints, points, np.ravel(cords[1, :]))
        interCords = np.asmatrix([interXCord,interYCord])
        return interCords

    def dist2cent(self):
        euclDist = LA.norm(self.normCords,2,0)
        return euclDist

