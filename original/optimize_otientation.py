import math
from scipy.spatial.transform import Rotation as R


def getXvector(r):
    xvector = r.apply([1, 0, 0])
    return xvector


def getYvector(r):
    yvector = r.apply([0, 1, 0])

    return yvector


def getZvector(r):
    zvector = r.apply([0, 0, 1])

    return zvector

def getOrientationType_simple(orientation, axis):
    # sbb = self.getBoundingBox()

    type = [0, 0, 0, 0, 0, 0]

    ex = orientation[0]
    ey = orientation[1]
    ez = orientation[2]

    r = R.from_euler('XYZ', [ex, ey, ez], degrees=True)
    # r= R.from_rotvec(o)

    x = getXvector(r)
    y = getYvector(r)
    z = getZvector(r)

    axes = [x, y, z]
    type[0] = 1

    axes.pop(axis)

    if abs(axes[0][1]) < abs(axes[1][1]):
        ref_axis = axes[0]
    else:
        ref_axis = axes[1]

    helper = ref_axis[1]
    if ref_axis[0] < 0:
        helper = -helper

    print(helper)

    type[3], type[4] = encode_orientation(helper)
    return type

def encode_orientation(value):
    if value > 0:
        pi_help = math.pi/math.sqrt(.5) * value
    else:
        pi_help = math.pi + math.pi/math.sqrt(.5) * (math.sqrt(.5) + value)

    cos_value = math.cos(pi_help)
    sin_value = math.sin(pi_help)

    return cos_value, sin_value
'''
def decode_orientation(cos_value, sin_value):
    asin_value = math.asin(sin_value)
    acos_value = math.acos(cos_value)

    if sin_value >= 0:
        return acos_value * math.sqrt(.5)/math.pi
    else:
        return - acos_value * math.sqrt(.5)/math.pi
        
'''

def decode_orientation(cos_value, sin_value):
    asin_value = math.asin(sin_value)
    neg_asin = math.asin(-sin_value)
    acos_value = math.acos(cos_value)

    if sin_value >= 0:
        if cos_value >= 0:
            mean = (acos_value + asin_value)/2
        else:
            mean = (acos_value + math.pi - asin_value)/2
    else:
        if cos_value < 0:
            mean = - (acos_value + math.pi - neg_asin)/2
        else:
            mean = - (acos_value + neg_asin) / 2

    return mean * math.sqrt(.5) / math.pi


type = getOrientationType_simple([0, 0, 10], 2)

print(type)
print(decode_orientation(type[3], type[4]))
