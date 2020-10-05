import sim
from scipy.spatial.transform import Rotation as R
import mathown
import math
import preferences
import csv


class Shape:
    def __init__(self, clientID, shape, number):  # probabilitySpace is a dictionary
        self.type = shape.__class__.__name__
        self.shapetype = shape
        if shape != "Cuboid" and shape != "Cylinder" and shape != "Sphere":
            print("false name")
        elif number == 0:
            self.name = shape + "#"
        else:
            name = shape + "{}#"
            self.name = name.format(number - 1)
        if shape == "Cuboid":
            self.shapetype_numbered = 0
        elif shape == "Sphere":
            self.shapetype_numbered = 1
        else:
            self.shapetype_numbered = 2
        self.clientID = clientID
        self.handle = self.getHandle()
        self.position = [0, 0, 0]
        self.velocity = [0, 0, 0]
        self.orientation = [0, 0, 0]  # rotation vector
        self.boundingBox = [0, 0, 0]
        self.boundingBoxWorld = [0, 0, 0]
        self.color = self.getColor()
        self.xvector = [0, 0, 0]  # x vector of internal coordinate system in relation to world coordinate system
        self.yvector = [0, 0, 0]  # y vector of internal coordinate system --
        self.zvector = [0, 0, 0]  # z vector of internal coordinate system --

    def getHandle(self):
        returnCode, self.handle = sim.simxGetObjectHandle(self.clientID, self.name, sim.simx_opmode_blocking)
        return self.handle

    # "get" functions
    def getPosition(self):
        returnCode, self.position = sim.simxGetObjectPosition(self.clientID, self.handle, -1, sim.simx_opmode_blocking)
        if returnCode == sim.simx_return_ok:
            return self.position
        else:
            return returnCode

    def getVelocity(self):
        returnCode, self.velocity = sim.simxGetObjectVelocity(self.clientID, self.handle, sim.simx_opmode_blocking)
        return self.velocity

    def getOrientation(self):
        returnCode, self.orientation = sim.simxGetObjectOrientation(self.clientID, self.handle, -1,
                                                                    sim.simx_opmode_blocking)
        return self.orientation

    def getXvector(self):
        r = R.from_rotvec(self.getOrientation())
        self.xvector = r.apply([1, 0, 0])
        return self.xvector

    def getYvector(self):
        r = R.from_rotvec(self.getOrientation())
        self.yvector = r.apply([0, 1, 0])
        return self.yvector

    def getZvector(self):
        r = R.from_rotvec(self.getOrientation())
        self.zvector = r.apply([0, 0, 1])
        return self.zvector

    def getOrientationFromCoordinateSystem(self, xvec, yvec, zvec):
        r = R.from_matrix([xvec, yvec, zvec])
        return r.as_rotvec()

    def getBoundingBox(self):
        returnCode, self.boundingBox[0] = sim.simxGetObjectFloatParameter(self.clientID, self.handle,
                                                                          sim.sim_objfloatparam_objbbox_max_x,
                                                                          sim.simx_opmode_blocking)
        self.boundingBox[0] *= 2
        returnCode, self.boundingBox[1] = sim.simxGetObjectFloatParameter(self.clientID, self.handle,
                                                                          sim.sim_objfloatparam_objbbox_max_y,
                                                                          sim.simx_opmode_blocking)
        self.boundingBox[1] *= 2
        returnCode, self.boundingBox[2] = sim.simxGetObjectFloatParameter(self.clientID, self.handle,
                                                                          sim.sim_objfloatparam_objbbox_max_z,
                                                                          sim.simx_opmode_blocking)
        self.boundingBox[2] *= 2
        return self.boundingBox

    def getBoundingBoxWorld(self):
        self.boundingBoxWorld[0] = abs(self.maxXvalue() - self.minXvalue())
        self.boundingBoxWorld[1] = abs(self.maxYvalue() - self.minYvalue())
        self.boundingBoxWorld[2] = abs(self.maxZvalue() - self.minZvalue())

        return self.boundingBoxWorld

    def getColor(self):
        object_handle = self.handle
        return_code, garbage1, self.color, garbage2, garbage3 = sim.simxCallScriptFunction(self.clientID, "ForScript",
                                                                                           sim.sim_scripttype_childscript,
                                                                                           "getShapeColor_function",
                                                                                           [object_handle], [], [],
                                                                                           bytearray(),
                                                                                           sim.simx_opmode_blocking)
        if return_code == sim.simx_return_ok:
            return self.color
        else:
            return return_code

    def setColor(self, r, g, b):
        object_handle = self.handle
        return_code, _, _, _, _ = sim.simxCallScriptFunction(self.clientID, "ForScript", sim.sim_scripttype_childscript,
                                                             "setShapeColor_function", [object_handle], [r, g, b], [],
                                                             bytearray(), sim.simx_opmode_blocking)

    def scaleShape(self, x, y, z):
        object_handle = self.handle
        return_code, _, _, _, _ = sim.simxCallScriptFunction(self.clientID, "ForScript", sim.sim_scripttype_childscript,
                                                             "setObjectShape_function", [object_handle], [x, y, z],
                                                             [], bytearray(), sim.simx_opmode_blocking)

    def getPositionRelation(self, shape):
        xRelation = shape.getPosition()[0] - self.getPosition()[0]
        if xRelation > 0:
            xRelation -= abs(shape.getBoundingBoxWorld()[0] * 0.5) + abs(self.getBoundingBoxWorld()[0] * 0.5)
        else:
            xRelation += abs(shape.getBoundingBoxWorld()[0] * 0.5) + abs(self.getBoundingBoxWorld()[0] * 0.5)
        yRelation = shape.getPosition()[1] - self.getPosition()[1]
        if yRelation > 0:
            yRelation -= abs(shape.getBoundingBoxWorld()[1] * 0.5) + abs(self.getBoundingBoxWorld()[1] * 0.5)
        else:
            yRelation += abs(shape.getBoundingBoxWorld()[1] * 0.5) + abs(self.getBoundingBoxWorld()[1] * 0.5)
        zRelation = shape.getPosition()[2] - self.getPosition()[2]
        if zRelation > 0:
            zRelation -= abs(shape.getBoundingBoxWorld()[2] * 0.5) + abs(self.getBoundingBoxWorld()[2] * 0.5)
        else:
            zRelation += abs(shape.getBoundingBoxWorld()[2] * 0.5) + abs(self.getBoundingBoxWorld()[2] * 0.5)
        relation = [xRelation, yRelation, zRelation]
        return relation

    # TO DO!!!
    def getRelationLikelihood(self, relationType, shape, agent):
        # probabilitySpace: {"preposition" -> [gaussX,gaussY,gaussZ]}
        if relationType in agent.probabilitySpace:
            gaussXr = agent.probabilitySpace[relationType][0]
            gaussXl = agent.probabilitySpace[relationType][1]
            gaussYb = agent.probabilitySpace[relationType][2]
            gaussYf = agent.probabilitySpace[relationType][3]
            gaussZh = agent.probabilitySpace[relationType][4]
            gaussZl = agent.probabilitySpace[relationType][5]

            positionRelation = self.getPositionRelation(shape)

            likelihood = (gaussXl.getProbability(positionRelation[0]) * 2 + gaussXr.getProbability(
                positionRelation[0]) * 2 + gaussYb.getProbability(positionRelation[1]) * 2 + gaussYf.getProbability(
                positionRelation[1]) * 2 + gaussZh.getProbability(positionRelation[2]) * 2 + gaussZl.getProbability(
                positionRelation[2]) * 2) / 6
            return likelihood
        else:
            print("false preposition")

    # "set" functions
    def setPosition(self, position):
        returnCode = sim.simxSetObjectPosition(self.clientID, self.handle, -1, position, sim.simx_opmode_oneshot)

    def setOrientation(self, orientation):
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.handle, -1, orientation, sim.simx_opmode_blocking)

    def setAtopCenter(self, object):
        self.setPosition([object.getPosition()[0], object.getPosition()[1],
                          object.getBoundingBoxWorld()[2] + self.getBoundingBoxWorld()[2] * 0.5])

    def setAtopVariable(self, object, leftright, frontback):
        self.setPosition([object.getPosition()[0] + leftright, object.getPosition()[1] + frontback,
                          object.getBoundingBoxWorld()[2] + self.getBoundingBoxWorld()[2] * 0.5])

    # helpers

    def rotateX(self, deg):
        r = R.from_euler('x', deg, degrees=True)
        lis = self.getOrientationFromCoordinateSystem(
            r.apply(self.getXvector()), r.apply(self.getYvector()), r.apply(self.getZvector()))
        self.setOrientation(lis.copy())
        return self.getOrientation()

    def rotateY(self, deg):
        r = R.from_euler('y', deg, degrees=True)
        lis = self.getOrientationFromCoordinateSystem(
            r.apply(self.getXvector()), r.apply(self.getYvector()), r.apply(self.getZvector()))
        self.setOrientation(lis.copy())
        return self.getOrientation()

    def rotateZ(self, deg):
        r = R.from_euler('z', deg, degrees=True)
        lis = self.getOrientationFromCoordinateSystem(
            r.apply(self.getXvector()), r.apply(self.getYvector()), r.apply(self.getZvector()))
        self.setOrientation(lis.copy())
        return self.getOrientation()

    def maxXvalue(self):
        x = self.getXvector() * self.getBoundingBox()[0]
        y = self.getYvector() * self.getBoundingBox()[1]
        z = self.getZvector() * self.getBoundingBox()[2]

        return max(x[0], y[0], z[0])

    def maxYvalue(self):
        x = self.getXvector() * self.getBoundingBox()[0]
        y = self.getYvector() * self.getBoundingBox()[1]
        z = self.getZvector() * self.getBoundingBox()[2]

        return max([x[1]], y[1], z[1])

    def maxZvalue(self):
        x = self.getXvector() * self.getBoundingBox()[0]
        y = self.getYvector() * self.getBoundingBox()[1]
        z = self.getZvector() * self.getBoundingBox()[2]

        return max([x[2]], y[2], z[2])

    def minXvalue(self):
        x = self.getXvector() * self.getBoundingBox()[0]
        y = self.getYvector() * self.getBoundingBox()[1]
        z = self.getZvector() * self.getBoundingBox()[2]

        return min([x[0]], y[0], z[0])

    def minYvalue(self):
        x = self.getXvector() * self.getBoundingBox()[0]
        y = self.getYvector() * self.getBoundingBox()[1]
        z = self.getZvector() * self.getBoundingBox()[2]

        return min([x[1]], y[1], z[1])

    def minZvalue(self):
        x = self.getXvector() * self.getBoundingBox()[0]
        y = self.getYvector() * self.getBoundingBox()[1]
        z = self.getZvector() * self.getBoundingBox()[2]

        return min([x[2]], y[2], z[2])
