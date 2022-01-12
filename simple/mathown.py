import math
import scipy.special as scsp
import numpy as np
import sim

class Gauss:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def standardNormal(self, value):
        sn = (math.pow(math.e, -math.pow(value, 2)) / math.sqrt(math.pi))
        return sn

    def computeNormalDistribution(self, value):
        gn = self.standardNormal(value) * (1 / math.sqrt(self.variance)) * (
                    (value - self.mean) / math.sqrt(self.variance))
        return gn

    def getZScore(self, value):
        z = (value - self.mean)/(math.sqrt(self.variance))
        return z

    def zToPValue(self, z):
        """From z-score return p-value."""
        return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))

    def getProbability(self, value):
        z = self.getZScore(value)
        p = self.zToPValue(z)
        if p > 0.5:
            return 1-p
        else:
            return p


def get_cos(sin):
    if sin > 1:
        sub = 1 - sin
        sin = 1 - sub
    if sin < -1:
        sub = -1 - sin
        sin = -1 + sub
    #if sin < 0:
    #    choice = np.array([math.asin(sin), - math.pi - math.asin(sin)])
    #else:
    #    choice = np.array([math.asin(sin), math.pi - math.asin(sin)])

    #x = np.random.choice(choice)
    x = math.asin(sin)

    return math.cos(x)
