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
            normalizedCords = centCords/(max(maxCords-minCords))
            return normalizedCords
        else:
            print('Singular point')
            return centCords

    def dist2cent(self):
        euclDist = LA.norm(self.normCords,2,0)
        return euclDist

