import numpy as np
from numpy import linalg as LA

class FeatureExtractor:
    def __init__(self,cords,lData = []):
        if not lData:
            lData.append(np.size(cords,1))
        self.lData = lData
        self.cords = cords #self.interpolateCords(cords, 100) TODO: Update inter points
        self.normCords = self.normalizeFull()


    def normalize(self):
        splittingPoints = [0] + (np.cumsum(self.lData) - 1).tolist()
        normCords = np.asmatrix(np.zeros(np.shape(self.cords)))
        for i in range(0,len(self.lData)):
            normCords[:,splittingPoints[i]:splittingPoints[i+1]] = self.normalizer(self.cords[:,splittingPoints[i]:splittingPoints[i+1]])
        return normCords

    def normalizeFull(self):
        splittingPoints = [0] + (np.cumsum(self.lData) - 1).tolist()
        normCords = np.asmatrix(np.zeros(np.shape(self.cords)))
        for i in range(0,len(self.lData)):
            normCords[:,splittingPoints[i]:splittingPoints[i+1]] = self.normalizerFull(self.cords[:,splittingPoints[i]:splittingPoints[i+1]])
        return normCords


    def normalizer(self,cords): # TODO: could try full normalization
        maxCords = cords.max(1)
        minCords = cords.min(1)
        diffCords = maxCords - minCords
        centCords = cords - (maxCords+minCords)/2
        if LA.norm(diffCords) != 0:
            normalizedCords = centCords/(max(maxCords-minCords))
            return normalizedCords
        else:
            print('Singular point')
            return centCords

    def normalizerFull(self,cords):
        maxCords = np.max(cords,1)
        minCords = np.min(cords,1)
        centCords = cords - (maxCords + minCords) / 2
        if LA.norm(maxCords - minCords) != 0:
            normalizedCords = np.divide(centCords,(maxCords - minCords))
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
        euclDist = np.asmatrix(LA.norm(self.normCords,2,0))
        return euclDist

    def angleOfMotion(self):
        #angleOfMotion = np.arctan2(self.normCords[0,:],self.normCords[1,:])
        diffCords = np.diff(self.cords)
        diffCordsExtended = np.concatenate((diffCords[:,0],diffCords),1)

        angleOfMotion = np.arctan2(diffCordsExtended[0, :], diffCordsExtended[1, :])
        return angleOfMotion
