import sim
from scipy.spatial.transform import Rotation as R
import mathown
import math
import preferences
import csv
import numpy as np
import d3g


class Shape:
    def __init__(self, clientID, shape, number):  # probabilitySpace is a dictionary
        self.type = shape.__class__.__name__
        self.shapetype = shape
        if shape != "Cuboid" and shape != "Cylinder" and shape != "Sphere" and shape != "Cone" and shape != "Pyramid":
            print("false name: " + shape)
            exit()
        elif number == 0:
            self.name = shape + "#"
        else:
            name = shape + "{}#"
            self.name = name.format(number - 1)
        if shape == "Cuboid":
            self.shapetype_numbered = 0
        elif shape == "Cylinder":
            self.shapetype_numbered = 1
        elif shape == "Sphere":
            self.shapetype_numbered = 2
        elif shape == "Cone":
            self.shapetype_numbered = 3
        elif shape == "Pyramid":
            self.shapetype_numbered = 4
        self.clientID = clientID
        self.handle = self.getHandle()
        self.position = [0, 0, 0]
        self.velocity = [0, 0, 0]
        self.orientation = [0, 0, 0, 0, 0, 0]  # sin and cos values of rotation vector
        self.boundingBox = [0, 0, 0]
        self.boundingBoxWorld = [0, 0, 0]
        #self.color = self.getColor()
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

    def getEdgePoints(self, position):
        x, y, z = self.getVectors()
        bb = self.getBoundingBox()

        x1 = self.vectorMultiplication(x, bb[0] * 0.5)
        x2 = self.vectorNegation(x1)
        y1 = self.vectorMultiplication(y, bb[1] * 0.5)
        y2 = self.vectorNegation(y1)
        z1 = self.vectorMultiplication(z, bb[2] * 0.5)
        z2 = self.vectorNegation(z1)

        p1 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x1, y1), z1), position)
        p2 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x1, y1), z2), position)
        p3 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x1, y2), z1), position)
        p4 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x1, y2), z2), position)
        p5 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x2, y1), z1), position)
        p6 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x2, y1), z2), position)
        p7 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x2, y2), z1), position)
        p8 = self.vectorAddition(self.vectorAddition(self.vectorAddition(x2, y2), z2), position)

        return p1, p2, p3, p4, p5, p6, p7, p8

    def getVelocity(self):
        returnCode, self.velocity = sim.simxGetObjectVelocity(self.clientID, self.handle, sim.simx_opmode_blocking)
        return self.velocity

    def getOrientation(self):
        if self.outOfBounds(self.getPosition()):
            self.orientation = [-1, -1, -1, -1, -1, -1]
        else:
            orientation = [0, 0, 0]
            returnCode, orientation = sim.simxGetObjectOrientation(self.clientID, self.handle, -1,
                                                                        sim.simx_opmode_blocking)

            self.orientation[0] = math.sin(orientation[0])
            self.orientation[1] = math.cos(orientation[0])
            self.orientation[2] = math.sin(orientation[1])
            self.orientation[3] = math.cos(orientation[1])
            self.orientation[4] = math.sin(orientation[2])
            self.orientation[5] = math.cos(orientation[2])
        return list(self.orientation)

    def getradianOrientation(self):
        returnCode, orientation = sim.simxGetObjectOrientation(self.clientID, self.handle, -1,
                                                                    sim.simx_opmode_blocking)
        return orientation

    def getXvector(self, r):

        self.xvector = r.apply([1, 0, 0])
        return self.xvector

    def getYvector(self, r):

        self.yvector = r.apply([0, 1, 0])

        return self.yvector

    def getZvector(self, r):

        self.zvector = r.apply([0, 0, 1])

        return self.zvector

    def getOrientationFromCoordinateSystem(self, xvec, yvec, zvec):
        r = R.from_matrix([xvec, yvec, zvec])
        return r.as_rotvec()

    def getOrientationType(self):
        sbb = self.getBoundingBox()
        o = self.getradianOrientation()

        if self.outOfBounds(self.getPosition()):
            return [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        ex = math.degrees(o[0])
        ey = math.degrees(o[1])
        ez = math.degrees(o[2])

        r= R.from_euler('XYZ', [ex, ey, ez], degrees=True)
        #r= R.from_rotvec(o)

        x = self.getXvector(r)
        y = self.getYvector(r)
        z = self.getZvector(r)

        flat, axis = self.isObjectflatonGround(x, y, z)

        if flat:
            axis_length = sbb[axis]
            rest = sbb.copy()
            rest.remove(axis_length)

            if self.isSmallest(axis_length, rest[0], rest[1]):
                type[0] = 1
            elif self.isLargest(axis_length, rest[0], rest[1]):
                type[1] = 1
            else:
                type[2] = 1

            larger = max(rest)
            index_l = sbb.index(larger)

            largest_axis = self.chooserightAxis(index_l, x, y, z)

            type[8] = math.asin(largest_axis[1])
            if largest_axis[0] < 0:
                type[8] = - type[8]
            type[9] = math.cos(type[8])
            type[8] = math.sin(type[8])

        else:
            edge, axis = self.isObjectonEdge(x, y, z)

            if edge:
                axis_length = sbb[axis]
                rest = sbb.copy()
                rest.remove(axis_length)

                if self.isSmallest(axis_length, rest[0], rest[1]):
                    type[4] = 1
                elif self.isLargest(axis_length, rest[0], rest[1]):
                    type[3] = 1
                else:
                    type[5] = 1

                l_axis = self.chooserightAxis(axis, x, y, z)

                type[8] = math.asin(l_axis[1])
                if l_axis[0] < 0:
                    type[8] = - type[8]
                type[9] = math.cos(type[8])
                type[8] = math.sin(type[8])

                larger = max(rest)
                index_l = sbb.index(larger)

                largest_axis = self.chooserightAxis(index_l, x, y, z)

                if (largest_axis[1] > 0 and largest_axis[2] < 0) or (largest_axis[1] < 0 and largest_axis[2] > 0):
                    type[10] = 1
                else:
                    type[10] = 0

            else:
                altX, altY, altZ = self.upwardfacingVectors(x, y, z)

                largest = max(sbb)
                index_l = sbb.index(largest)
                rest = sbb.copy()
                rest.remove(largest)
                medium = max(rest)
                index_m = sbb.index(medium)
                rest.remove(medium)
                smallest = rest[0]
                index_s = sbb.index(smallest)

                largest_axis = self.chooserightAxis(index_l, altX, altY, altZ)
                medium_axis = self.chooserightAxis(index_m, altX, altY, altZ)
                smallest_axis = self.chooserightAxis(index_s, altX, altY, altZ)

                #which sides are forward facing?
                if largest_axis[1] < 0:
                    type[8] = 1
                if smallest_axis[1] < 0:
                    type[9] = 1
                if medium_axis[1] < 0:
                    type[10] = 1

                #which of the two possible edge types applies?
                m = (largest_axis[1] - medium_axis[1])/(largest_axis[0] - medium_axis[0])
                c = largest_axis[1] - m * largest_axis[0]

                y_s = m * smallest_axis[0] + c

                if largest_axis[0] > medium_axis[0]:
                    if y_s < smallest_axis[1]:
                        type[7] = 1
                    else:
                        type[6] = 1
                else:
                    if y_s < smallest_axis[1]:
                        type[6] = 1
                    else:
                        type[7] = 1

        return type


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

    def getType(self):
        return self.shapetype_numbered

    def getVectors(self):
        o = self.getradianOrientation()

        ex = math.degrees(o[0])
        ey = math.degrees(o[1])
        ez = math.degrees(o[2])

        r = R.from_euler('XYZ', [ex, ey, ez], degrees=True)

        x = self.getXvector(r)
        y = self.getYvector(r)
        z = self.getZvector(r)

        return x, y, z

    def getBoundingBoxWorld(self):
        sbb = self.getBoundingBox()

        x, y, z = self.getVectors()

        maxXvalue = self.maxXvalue(sbb, x, y, z)
        maxYvalue = self.maxYvalue(sbb, x, y, z)
        maxZvalue = self.maxZvalue(sbb, x, y, z)
        minXvalue = self.minXvalue(sbb, x, y, z)
        minYvalue = self.minYvalue(sbb, x, y, z)
        minZvalue = self.minZvalue(sbb, x, y, z)

        self.boundingBoxWorld[0] = abs(maxXvalue - minXvalue)
        self.boundingBoxWorld[1] = abs(maxYvalue - minYvalue)
        self.boundingBoxWorld[2] = abs(maxZvalue - minZvalue)

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

    def getRelativePosition(self, object, position, first):
        if first:
            return [-1, -1, -1, -1, -1, -1, -1, -1]

        obb = object.getBoundingBoxWorld()
        op = object.getPosition()
        sbb = self.getBoundingBoxWorld()
        if self.sameObject(op, position):
            return [0, 0, 0, 0, 0, 0, 0, 0]

        else:
            variable = [0, 0, 0]
            if self.tooFarAway(op, position):
                return [-1, -1, -1, -1, -1, -1, -1, -1]

            if self.isOnTopOf(obb, op, sbb, position):
                variable[0] = position[0] - op[0]
                variable[1] = position[1] - op[1]
                variable[2] = position[2] - (obb[2] + 0.5 * sbb[2])
                return [1, 0, 0, 0, 0, variable[0], variable[1], variable[2]]
            elif self.isLeftOf(obb, op, sbb, position):
                variable[0] = position[0] - (op[0] - 0.5)
                variable[1] = position[1] - op[1]
                variable[2] = position[2] - sbb[2] * 0.5
                return [0, 1, 0, 0, 0, variable[0], variable[1], variable[2]]
            elif self.isRightOf(obb, op, sbb, position):
                variable[0] = position[0] - (op[0] + 0.5)
                variable[1] = position[1] - op[1]
                variable[2] = position[2] - sbb[2] * 0.5
                return [0, 0, 1, 0, 0, variable[0], variable[1], variable[2]]
            elif self.isInFrontOf(obb, op, sbb, position):
                variable[0] = position[0] - op[0]
                variable[1] = position[1] - (op[1] - 0.5)
                variable[2] = position[2] - sbb[2] * 0.5
                return [0, 0, 0, 1, 0, variable[0], variable[1], variable[2]]
            elif self.isBehind(obb, op, sbb, position):
                variable[0] = position[0] - op[0]
                variable[1] = position[1] - (op[1] + 0.5)
                variable[2] = position[2] - sbb[2] * 0.5
                return [0, 0, 0, 0, 1, variable[0], variable[1], variable[2]]
            else:
                print("error: no positional relation")
                return [-1, -1, -1, -1, -1, -1, -1, -1]

    # "set" functions
    def setPosition(self, position, shapelist):
        bb = self.getBoundingBoxWorld()
        #inSpaceOf, shape = self.inSpaceOf(position, shapelist, bb)
        collision, shape = self.collision(position, shapelist, bb)
        belowGround = self.below_ground(position, bb)
        outOfBounds = self.outOfBounds(position)
        while collision or belowGround or outOfBounds:
        #while belowGround or outOfBounds:
            if outOfBounds:
                print("out of bounds")
                print(position)
                if position[0] < -2.5:
                    position[0] = -2.4
                elif position[0] > 2.5:
                    position[0] = 2.4
                elif position[1] < -2.5:
                    position[1] = -2.4
                elif position[1] > 2.5:
                    position[1] = 2.4
            if collision:
                print("collision")
                print(position)
                sbb = shape.getBoundingBoxWorld()
                position = [position[0], position[1], position[2] + sbb[2]]
            if belowGround:
                print("below ground")
                print(position)
                print(bb)
                position = [position[0], position[1], bb[2]*0.5]

            collision, shape = self.inSpaceOf(position, shapelist, bb)
            belowGround = self.below_ground(position, bb)
            outOfBounds = self.outOfBounds(position)

        returnCode = sim.simxSetObjectPosition(self.clientID, self.handle, -1, position, sim.simx_opmode_blocking)

        return position

    def set_relative_position(self, position, object, leftright, frontback, shapelist):
        position = position.to(device="cpu")
        number = np.argmax(position)
        if self.sameObject(object.getPosition(), self.getPosition()):
            self.moveTo(0, 0, shapelist)
        elif number == 0:
            if position[0] == 0:
                self.moveTo(0, 0, shapelist)
            else:
                self.setAtopVariable(object, leftright, frontback, shapelist)
        elif number == 1:
            self.moveLeftOfVariable(object, leftright, frontback, shapelist)
        elif number == 2:
            self.moveRightOfVariable(object, leftright, frontback, shapelist)
        elif number == 3:
            self.moveInFrontOfVariable(object, leftright, frontback, shapelist)
        elif number == 4:
            if position[4] == 0:
                self.moveTo(0, 0, shapelist)
            else:
                self.moveBehindVariable(object, leftright, frontback, shapelist)

    def setPosition_eval_from_relativePosition(self, position, object, leftright, frontback, updown, shapelist):
        sbb = self.getBoundingBoxWorld()
        number = np.argmax(position)
        if number == 0:
            if position[0] == 0:
                self.setPosition_eval([0, 0, sbb[2] * 0.5], shapelist)
            else:
                self.setAtopVariable(object, leftright, frontback, shapelist, updown, True)
        elif number == 1:
            self.moveLeftOfVariable(object, leftright, frontback, shapelist, updown, True)
        elif number == 2:
            self.moveRightOfVariable(object, leftright, frontback, shapelist, updown, True)
        elif number == 3:
            self.moveInFrontOfVariable(object, leftright, frontback, shapelist, updown, True)
        elif number == 4:
            if position[4] == 0:
                self.setPosition_eval([0, 0, sbb[2] * 0.5], shapelist)
            else:
                self.moveBehindVariable(object, leftright, frontback, shapelist, updown, True)

    def setPosition_eval(self, position, shapelist):
        returnCode = sim.simxSetObjectPosition(self.clientID, self.handle, -1, position, sim.simx_opmode_blocking)

    def setOrientation(self, orientation):
        a_cos = [0, 0, 0]
        a_sin = [0, 0, 0]
        a_sin[0] = math.asin(orientation[0])
        a_cos[0] = math.acos(orientation[1])
        a_sin[1] = math.asin(orientation[2])
        a_cos[1] = math.acos(orientation[3])
        a_sin[2] = math.asin(orientation[4])
        a_cos[2] = math.acos(orientation[5])

        rad_or = [0, 0, 0]

        for i in range(3):
            if orientation[2*i] >= 0 and orientation[2*i+1] <= 0:
                rad_or[i] = a_cos[i]
            elif orientation[2*i] <= 0 and orientation[2*i+1] <= 0:
                rad_or[i] = - a_cos[i]
            elif orientation[2*i] <= 0 and orientation[2*i+1] >= 0:
                rad_or[i] = a_sin[i]
            elif orientation[2*i] >= 0 and orientation[2*i+1] >= 0:
                rad_or[i] = a_sin[i]
            else:
                print("Error: no orientation")

        returnCode = sim.simxSetObjectOrientation(self.clientID, self.handle, -1, rad_or, sim.simx_opmode_blocking)

    def setVisualOrientation(self, orientation):
        sbb = self.getBoundingBox()
        l = max(sbb)
        l_index = sbb.index(l)
        s = min(sbb)
        s_index = sbb.index(s)

        if orientation[0] == 1 or orientation[1] == 1 or orientation[2] == 1:
            self.setPlaneOrientation(orientation, l_index, s_index)
        elif orientation[3] == 1 or orientation[4] == 1 or orientation[5] == 1:
            self.setEdgeOrientation(orientation, l_index, s_index)
        else:
            self.setPointOrientation(orientation, l_index, s_index)

    def rotatePlaneOrientation(self, upaxis, frontaxis, rotation):
        new_or = [0, 0, 0]
        if upaxis == 0:
            new_or[0] = math.pi/2
            new_or[2] = math.pi/2
            if frontaxis == 1: #largest axis parallel to world x-axis plus saved rotation
                new_or[1] = rotation
            else:
                new_or[1] = math.pi/2 + rotation
        elif upaxis == 1:
            new_or[0] = math.pi/2
            new_or[2] = 0
            if frontaxis == 0:
                new_or[1] = rotation
            else:
                new_or[1] = math.pi/2 + rotation
        else:
            new_or[0] = 0
            new_or[1] = 0
            if frontaxis == 0:
                new_or[2] = rotation
            else:
                new_or[2] = -math.pi/2 + rotation
        return new_or

    def rotateEdgeOrientation(self, edgeaxis, largeaxis, rotation, forwardfacing):
        new_or = [0, 0, 0]
        if edgeaxis == 0:
            new_or[1] = 0
            if largeaxis == 1:
                if forwardfacing:
                    new_or[0] = math.pi - math.pi/4
                    new_or[2] = rotation
                else:
                    new_or[0] = math.pi/4
                    new_or[2] = rotation
            else:
                if forwardfacing:
                    new_or[0] = math.pi/4
                    new_or[2] = rotation
                else:
                    new_or[0] = math.pi - math.pi/4
                    new_or[2] = rotation
        elif edgeaxis == 1:
            new_or[0] = 0
            if largeaxis == 0:
                if forwardfacing:
                    new_or[1] = - math.pi/4
                    new_or[2] = - math.pi/2 + rotation
                else:
                    new_or[1] = - math.pi/4
                    new_or[2] = math.pi/2 + rotation
            else:
                if not forwardfacing:
                    new_or[1] = - math.pi/4
                    new_or[2] = - math.pi/2 + rotation
                else:
                    new_or[1] = - math.pi/4
                    new_or[2] = math.pi/2 + rotation
        else:
            new_or[0] = math.pi/2
            if largeaxis == 0:
                if not forwardfacing:
                    new_or[1] = - math.pi/4
                    new_or[2] = math.pi/2 + rotation
                else:
                    new_or[1] = - math.pi + math.pi/4
                    new_or[2] = math.pi/2 + rotation
            else:
                if forwardfacing:
                    new_or[1] = - math.pi/4
                    new_or[2] = math.pi/2 + rotation
                else:
                    new_or[1] = + math.pi/4
                    new_or[2] = math.pi/2 + rotation
        return new_or

    def setEdgeOrientation(self, orientation, l_index, s_index):
        m_index = [0, 1, 2]
        m_index.remove(l_index)
        m_index.remove(s_index)
        m_index = int(m_index[0])
        rotation = math.asin(orientation[8])
        forwardfacing = False
        if orientation[10] == 1:
            forwardfacing = True

        if orientation[3] == 1:
            new_or = self.rotateEdgeOrientation(l_index, m_index, rotation, forwardfacing)
        elif orientation[4] == 1:
            new_or = self.rotateEdgeOrientation(s_index, l_index, rotation, forwardfacing)
        else:
            new_or = self.rotateEdgeOrientation(m_index, l_index, rotation, forwardfacing)

        adjusted_or = self.rotationfromWorldAxis(new_or[0], new_or[1], new_or[2])

        self.setradianOrientation(adjusted_or)

    def setPointOrientation(self, orientation, l_index, s_index):
        m_index = [0, 1, 2]
        m_index.remove(l_index)
        m_index.remove(s_index)
        m_index = int(m_index[0])

        frontfacing1 = False
        frontfacing2 = False

        firstOption = False
        if orientation[6] == 1:
            firstOption = True

        tilted = False
        if s_index == 0:
            if l_index == 2:
                tilted = True
        elif s_index == 1:
            if l_index == 0:
                tilted = True
        else:
            if l_index == 1:
                tilted = True

        if not tilted and firstOption:
            simple = True
        if tilted and firstOption:
            simple = False
        if not tilted and not firstOption:
            simple = False
        if tilted and not firstOption:
            simple = True

        counter = 0
        if orientation[8] == 1:
            frontfacing1 = l_index
            counter += 1
        if orientation[9] == 1:
            if not frontfacing1:
                frontfacing1 = s_index
            else: frontfacing2 = s_index
            counter +=1
        if orientation[10] == 1:
            if not frontfacing1:
                frontfacing1 = m_index
            else: frontfacing2 = m_index
            counter +=1

        new_or = self.rotatePointOrientation(counter, frontfacing1, frontfacing2, simple)

        adjusted_or = self.rotationfromWorldAxis(new_or[0], new_or[1], new_or[2])

        self.setradianOrientation(adjusted_or)

    def rotatePointOrientation(self, num_front, frontfacing1, frontfacing2, simple):
        new_or = [0, 0, 0]
        if simple:
            new_or[0] = math.pi/4
            new_or[1] = - math.pi/4
            if num_front == 1:
                if frontfacing1 == 0:
                    rotate = - math.pi/2
                elif frontfacing1 == 1:
                    rotate = math.pi - math.pi/4
                else:
                    rotate = math.pi/4
            else:
                if frontfacing1 != 0 and frontfacing2 != 0:
                    rotate = math.pi/2
                elif frontfacing1 != 1 and frontfacing2 != 1:
                    rotate = - math.pi/4
                else:
                    rotate = math.pi + math.pi/4

            new_or[2] = rotate
        else:
            new_or[0] = math.pi/4
            new_or[1] = math.pi/4
            if num_front == 1:
                if frontfacing1 == 0:
                    rotate = math.pi/2
                elif frontfacing1 == 1:
                    rotate = math.pi + math.pi/4
                else:
                    rotate = - math.pi/4
            else:
                if frontfacing1 != 0 and frontfacing2 != 0:
                    rotate = - math.pi/2
                elif frontfacing1 != 1 and frontfacing2 != 1:
                    rotate = math.pi/4
                else:
                    rotate = math.pi - math.pi/4

            new_or[2] = rotate

        return new_or

    def setPlaneOrientation(self, orientation, l_index, s_index):
        m_index = [0, 1, 2]
        m_index.remove(l_index)
        m_index.remove(s_index)
        m_index = int(m_index[0])
        rotation = math.asin(orientation[8])

        if orientation[0] == 1: #smallest axis orthogonal to bottom plane
            new_or = self.rotatePlaneOrientation(s_index, l_index, rotation)
        elif orientation[1] == 1: # largest axis orthogonal to bottom plane
            new_or = self.rotatePlaneOrientation(l_index, m_index, rotation)
        else:
            new_or = self.rotatePlaneOrientation(m_index, l_index, rotation)

        self.setradianOrientation(new_or)

    def setradianOrientation(self, orientation):
        returnCode = sim.simxSetObjectOrientation(self.clientID, self.handle, -1, orientation, sim.simx_opmode_blocking)

    def setAtopCenter(self, object, shapelist):
        op = object.getPosition()
        obb = object.getBoundingBoxWorld()

        position = [op[0], op[1], op[2] + obb[2] * 0.5 + 0.3]
        position = self.setPosition(position, shapelist)

        return list(position)

    def setAtopVariable(self, object, leftright, frontback, shapelist, updown=0, eval=False):
        op = object.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0] + leftright, op[1] + frontback, op[2] + obb[2] * 0.5 + sbb[2] * 0.5 + updown]
        if eval:
            self.setPosition_eval(position, shapelist)
        else:
            position = self.setPosition(position, shapelist)

        return list(position)

    def moveTo(self, x, y, shapelist):
        position = [x, y, self.getBoundingBoxWorld()[2] * 0.5]
        position = self.setPosition(position, shapelist)

        return list(position)

    def moveLeftOf(self, object, shapelist):
        op = object.getPosition()
        sbb = self.getBoundingBoxWorld()

        position = [op[0] - 0.5, op[1], sbb[2] * 0.5]
        self.setPosition(position, shapelist)

        return list(position)

    def moveLeftOfVariable(self, object, leftright, frontback, shapelist, updown=0.0, eval=False):
        op = object.getPosition()
        sp = self.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0] - 0.5 + leftright, op[1] + frontback, sbb[2] * 0.5 + updown]
        if eval:
            self.setPosition_eval(position, shapelist)
        else:
            position = self.setPosition(position, shapelist)

        return list(position)

    def moveRightOfVariable(self, object, leftright, frontback, shapelist, updown=0.0, eval=False):

        op = object.getPosition()
        sp = self.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0] + 0.5 + leftright, op[1] + frontback, sbb[2] * 0.5 + updown]
        if eval:
            self.setPosition_eval(position, shapelist)
        else:
            position = self.setPosition(position, shapelist)

        return list(position)

    def moveInFrontOfVariable(self, object, leftright, frontback, shapelist, updown=0, eval=False):

        op = object.getPosition()
        sp = self.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0] + leftright, op[1] - 0.5 + frontback, sbb[2] * 0.5 + updown]
        if eval:
            self.setPosition_eval(position, shapelist)
        else:
            position = self.setPosition(position, shapelist)

        return list(position)

    def moveBehindVariable(self, object, leftright, frontback, shapelist, updown=0, eval=False):

        op = object.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0] + leftright, op[1] + 0.5 + frontback, sbb[2] * 0.5 + updown]
        if eval:
            self.setPosition_eval(position, shapelist)
        else:
            position = self.setPosition(position, shapelist)

        return list(position)

    def moveRightOf(self, object, shapelist):
        op = object.getPosition()
        sp = self.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0] + 0.5, op[1], sp[2]]
        self.setPosition(position, shapelist)

        return list(position)

    def moveInFrontOf(self, object, shapelist):
        op = object.getPosition()
        sp = self.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0], op[1] - 0.5, sp[2]]

        self.setPosition(position, shapelist)

        return list(position)

    def moveBehind(self, object, shapelist):
        op = object.getPosition()
        sp = self.getPosition()
        obb = object.getBoundingBoxWorld()
        sbb = self.getBoundingBoxWorld()

        position = [op[0], op[1] + 0.5, sp[2]]

        self.setPosition(position, shapelist)

        return list(position)

    def performRandomAction(self, object, number, shapeslist):
        if number == 0:
            position = self.moveLeftOf(object, shapeslist)
        elif number == 1:
            position = self.moveRightOf(object, shapeslist)
        elif number == 2:
            position = self.moveInFrontOf(object, shapeslist)
        elif number == 3:
            position = self.moveBehind(object, shapeslist)
        else:
            position = self.setAtopCenter(object, shapeslist)

        return position

    def performRandomActionVariable(self, object, number, leftright, frontback, shapeslist):
        if number == 0:
            position = self.moveLeftOfVariable(object, leftright, frontback, shapeslist)
        elif number == 1:
            position = self.moveRightOfVariable(object, leftright, frontback, shapeslist)
        elif number == 2:
            position = self.moveInFrontOfVariable(object, leftright, frontback, shapeslist)
        elif number == 3:
            position = self.moveBehindVariable(object, leftright, frontback, shapeslist)
        else:
            position = self.setAtopVariable(object, leftright, frontback, shapeslist)

        position = self.getRelativePosition(object, position, first=False)

        return position

    def turnOriginalWayUp(self):
        self.setOrientation([0, 1, 0, 1, self.getOrientation()[4], self.getOrientation()[5]])

    # helpers
    def collision(self, position, shapelist, BB):
        p1, p2, p3, p4, p5, p6, p7, p8 = self.getEdgePoints(position)
        points = [p1, p2, p3, p4, p5, p6, p7, p8]

        plane1 = self.getPlane(p1, p5)
        plane2 = self.getPlane(p1, p2)
        plane3 = self.getPlane(p1, p3)

        plane11 = self.getPlane(p8, p4)
        plane21 = self.getPlane(p8, p7)
        plane31 = self.getPlane(p8, p6)

        planes = [[plane1, plane2, plane3, plane11, plane21, plane31], [[p1, p2, p3], [p1, p3, p5], [p1, p2, p5],
                                                                        [p8, p7, p6], [p8, p4, p6], [p8, p4, p7]]]
        #print(planes[0])
        line1 = self.getLine(p1, p2)
        line2 = self.getLine(p1, p5)
        line3 = self.getLine(p5, p6)
        line4 = self.getLine(p2, p6)
        line5 = self.getLine(p3, p4)
        line6 = self.getLine(p3, p7)
        line7 = self.getLine(p7, p8)
        line8 = self.getLine(p4, p8)
        line9 = self.getLine(p5, p7)
        line10 = self.getLine(p1, p6)
        line11 = self.getLine(p2, p4)
        line12 = self.getLine(p6, p8)

        lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12]
        #print(lines)
        for obj in shapelist:
            op1, op2, op3, op4, op5, op6, op7, op8 = obj.getEdgePoints(obj.getPosition())
            opoints = [op1, op2, op3, op4, op5, op6, op7, op8]

            oplane1 = self.getPlane(op1, op5)
            oplane2 = self.getPlane(op1, op2)
            oplane3 = self.getPlane(op1, op3)

            oplane11 = self.getPlane(op8, op4)
            oplane21 = self.getPlane(op8, op7)
            oplane31 = self.getPlane(op8, op6)

            oplanes = [[oplane1, oplane2, oplane3, oplane11, oplane21, oplane31], [[op1, op2, op3], [op1, op3, op5], [op1, op2, op5],
                                                                            [op8, op7, op6], [op8, op4, op6], [op8, op4, op7]]]

            oline1 = self.getLine(op1, op2)
            oline2 = self.getLine(op1, op5)
            oline3 = self.getLine(op5, op6)
            oline4 = self.getLine(op2, op6)
            oline5 = self.getLine(op3, op4)
            oline6 = self.getLine(op3, op7)
            oline7 = self.getLine(op7, op8)
            oline8 = self.getLine(op4, op8)
            oline9 = self.getLine(op5, op7)
            oline10 = self.getLine(op1, op6)
            oline11 = self.getLine(op2, op4)
            oline12 = self.getLine(op6, op8)

            olines = [oline1, oline2, oline3, oline4, oline5, oline6, oline7, oline8, oline9, oline10, oline11, oline12]

            #print(1)
            for line in olines:
                for i in range(6):
                    parallel, intersection, r = self.intersect(planes[0][i], line)
                    if not parallel:
                        # test if intersection lies between the edge points
                        if 0 <= r <= 1:
                            # test if intersection lies in rectangle of plane
                            if self.inRectangle(intersection, planes[1][i]):
                                return True, obj
            #print(2)
            for line in lines:
                for i in range(6):
                    parallel, intersection, r = self.intersect(oplanes[0][i], line)
                    if not parallel:
                        # test if intersection lies between the edge points
                        # if self.inLineSegment(intersection, line):
                        if 0 <= r <= 1:
                            #print("points 1 " + str(points))
                            #print("points 2 " + str(opoints))
                            #print("point " + str(intersection))
                            #print("line " + str(line))
                            #print("plane " + str(oplanes[0][i]))
                            #print("in segment")
                            # test if intersection lies in rectangle of plane
                            if self.inRectangle(intersection, oplanes[1][i]):
                                #print("Rectangle points " + str(oplanes[1][i]))
                                #print("in rectangle")
                                return True, obj

        return False, self

    def get_other_shape_id(self, idx):
        if idx == 0:
            return 1
        else:
            return 0

    def getPlane(self, p1, p2): #Normalenform
        n = self.vectorSubtraction(p2, p1)
        no = self.vectorDivision(n, self.vectorLength(n))

        return [no, p1]

    def inRectangle(self, s, points):
        a = self.vectorSubtraction(points[1], points[0])
        b = self.vectorSubtraction(points[2], points[0])
        p = points[0]

        if a[0] != 0:
            e = s[0] / a[0]
            c = p[0] / a[0]
            d = b[0] / a[0]
            if b[1] - d * a[1] != 0:
                v = (s[1] - p[1] - e * a[1] + c * a[1]) / (b[1] - d * a[1])
                u = e - c - v * d
            else:
                v = (s[2] - p[2] - e * a[2] + c * a[2]) / (b[2] - d * a[2])
                u = e - c - v * d
        elif a[1] != 0:
            e = s[1] / a[1]
            c = p[1] / a[1]
            d = b[1] / a[1]
            if b[2] - d * a[2] != 0:
                v = (s[2] - p[2] - e * a[2] + c * a[2]) / (b[2] - d * a[2])
                u = e - c - v * d
            else:
                v = (s[0] - p[0] - e * a[0] + c * a[0]) / (b[0] - d * a[0])
                u = e - c - v * d
        else:
            e = s[2] / a[2]
            c = p[2] / a[2]
            d = b[2] / a[2]
            if b[1] - d * a[1] != 0:
                v = (s[1] - p[1] - e * a[1] + c * a[1]) / (b[1] - d * a[1])
                u = e - c - v * d
            else:
                v = (s[0] - p[0] - e * a[0] + c * a[0]) / (b[0] - d * a[0])
                u = e - c - v * d

        return 0 <= u <= 1 and 0 <= v <= 1

    def scalarProduct(self, p, n):
        return p[0]*n[0] + p[1]*n[1] + p[2]*n[2]

    def vectorLength(self, p):
        return math.sqrt(math.pow(p[0], 2) + math.pow(p[1], 2) + math.pow(p[2], 2))

    def vectorDivision(self, p, n):
        return [p[0]/n, p[1]/n, p[2]/n]

    def vectorNegation(self, p):
        return [-p[0], -p[1], -p[2]]

    def vectorSubtraction(self, p1, p2):
        return [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]

    def vectorAddition(self, p1, p2):
        return [p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]]

    def vectorMultiplication(self, p, scalar):
        return [p[0] * scalar, p[1] * scalar, p[2] * scalar]

    def getDistanceFromPlane(self, plane, p):
        return self.scalarProduct(plane[0], self.vectorSubtraction(p, plane[1]))

    def getLine(self, p1, p2):
        return [p1, self.vectorSubtraction(p2, p1)]

    def intersect(self, plane, line):
        if self.scalarProduct(plane[0], line[1]) == 0:
            return True, [], 0

        r = (self.scalarProduct(plane[0], plane[1]) - self.scalarProduct(plane[0], line[0])) / self.scalarProduct(plane[0], line[1])
        p = self.vectorAddition(line[0], self.vectorMultiplication(line[1], r))
        return False, p, r

    def inLineSegment(self, p, line):
        if line[1][0] != 0:
            r = (p[0] - line[0][0]) / line[1][0]
        elif line[1][1] != 0:
            r = (p[1] - line[0][1]) / line[1][1]
        else:
            r = (p[2] - line[0][2]) / line[1][2]
        return 0 <= r <= 1

    def isObjectflatonGround(self, x, y, z):
        if 0.99 < x[2] < 1.01 or -0.99 > x[2] > -1.01:
            return True, 0
        elif 0.99 < y[2] < 1.01 or -0.99 > y[2] > -1.01:
            return True, 1
        elif 0.99 < z[2] < 1.01 or -0.99 > z[2] > -1.01:
            return True, 2
        else:
            return False, -1

    def isObjectonEdge(self, x, y, z):
        if -0.05 < x[2] < 0.05:
            return True, 0
        elif -0.05 < y[2] < 0.05:
            return True, 1
        elif -0.05 < z[2] < 0.05:
            return True, 2
        else:
            return False, -1

    def isLargest(self, a, b, c):
        return a > b and a > c

    def isSmallest(self, a, b, c):
        return a < b and a < c

    def isMedium(self, a, b, c):
        return (b < a < c) or (c < a < b)

    def below_ground(self, position, bb):
        z = position[2]
        bbz = bb[2] * 0.5

        return z-bbz < 0

    def outOfBounds(self, position):
        return not (2.5 > position[0] > -2.5 and 2.5 > position[1] > -2.5)

    def inSpaceOf(self, position, shapelist, BB):

        x1 = [position[0] - BB[0] * 0.5,
              position[0] + BB[0] * 0.5]
        y1 = [position[1] - BB[1] * 0.5,
              position[1] + BB[1] * 0.5]
        z1 = [position[2] - BB[2] * 0.5,
              position[2] + BB[2] * 0.5]

        for shape in shapelist:
            sbb = shape.getBoundingBoxWorld()
            sp = shape.getPosition()

            x2 = [sp[0] - sbb[0] * 0.5,
                  sp[0] + sbb[0] * 0.5]
            y2 = [sp[1] - sbb[1] * 0.5,
                  sp[1] + sbb[1] * 0.5]
            z2 = [sp[2] - sbb[2] * 0.5,
                  sp[2] + sbb[2] * 0.5]
            if min(x1[1], x2[1]) - max(x1[0], x2[0]) > 0:
                if min(y1[1], y2[1]) - max(y1[0], y2[0]) > 0:
                    if min(z1[1], z2[1]) - max(z1[0], z2[0]) > 0:
                        return True, shape

        return False, self

    def isOnTopOf(self, obb, op, sbb, position):
        ontop = False

        if abs(op[0] - position[0]) <= obb[0] * 0.5 and abs(op[1] - position[1]) <= obb[1] * 0.5 and position[2] > op[2] + 0.5 * obb[2] + 0.4 * sbb[2]:
            ontop = True

        return ontop

    def notTouching(self, obb, op, sbb, position):
        x1 = [position[0] - sbb[0] * 0.5,
              position[0] + sbb[0] * 0.5]
        y1 = [position[1] - sbb[1] * 0.5,
              position[1] + sbb[1] * 0.5]

        x2 = [op[0] - obb[0] * 0.5,
              op[0] + sbb[0] * 0.5]
        y2 = [op[1] - obb[1] * 0.5,
              op[1] + obb[1] * 0.5]
        if min(x1[1], x2[1]) - max(x1[0], x2[0]) > 0:
            if min(y1[1], y2[1]) - max(y1[0], y2[0]) > 0:
                return False

        return True


    def isLeftOf(self, obb, op, sbb, position):
        left = False

        #compute the diagonal straights going through the center of the reference object as borders
        c = op[1] - op[0]
        b = op[1] + op[0]

        y1 = position[0] + c
        y2 = - position[0] + b

        if position[0] < op[0] and y1 <= position[1] <= y2:
            left = True

        return left

    def tooFarAway(self, op, position):
        toofar = False
        if abs(op[0] - position[0]) > 2.5 or abs(op[1] - position[1]) > 2.5:
            toofar = True

        return toofar

    def isRightOf(self, obb, op, sbb, position):
        right = False

        #compute the diagonal straights going through the center of the reference object as borders
        c = op[1] - op[0]
        b = op[1] + op[0]

        y1 = position[0] + c
        y2 = - position[0] + b

        if position[0] > op[0] and y1 >= position[1] >= y2:
            right = True

        return right

    def isInFrontOf(self, obb, op, sbb, position):
        infront = False

        #compute the diagonal straights going through the center of the reference object as borders
        c = op[1] - op[0]
        b = op[1] + op[0]

        y1 = position[0] + c
        y2 = - position[0] + b

        if position[1] < op[1] and position[1] <= y2 and position[1] <= y1:
            infront = True

        return infront

    def sameObject(self, op, position):
        if abs(op[0] - position[0]) < 0.005 and abs(op[1] - position[1]) < 0.005 and abs(op[2] - position[2]) < 0.005:
            return True

        return False

    def isBehind(self, obb, op, sbb, position):
        behind = False

        #compute the diagonal straights going through the center of the reference object as borders
        c = op[1] - op[0]
        b = op[1] + op[0]

        y1 = position[0] + c
        y2 = - position[0] + b

        if position[1] > op[1] and position[1] >= y1 and position[1] >= y2:
            behind = True

        return behind

    def rotateX(self, deg):
        r = R.from_euler('x', deg, degrees=True)
        lis = self.getOrientationFromCoordinateSystem(self.getXvector(r), self.getYvector(r), self.getZvector(r))
        self.setradianOrientation(lis.copy())
        return self.getOrientation()

    def rotateY(self, deg):
        r = R.from_euler('y', deg, degrees=True)
        lis = self.getOrientationFromCoordinateSystem(self.getXvector(r), self.getYvector(r), self.getZvector(r))
        self.setradianOrientation(lis.copy())
        return self.getOrientation()

    def rotateZ(self, deg):
        r = R.from_euler('z', deg, degrees=True)
        lis = self.getOrientationFromCoordinateSystem(self.getXvector(r), self.getYvector(r), self.getZvector(r))
        self.setradianOrientation(lis.copy())
        return self.getOrientation()

    def maxXvalue(self, sbb, x1, y1, z1):

        x = x1 * sbb[0]
        y = y1 * sbb[1]
        z = z1 * sbb[2]

        return max(0, x[0], y[0], z[0], x[0] + y[0], x[0] + z[0], y[0] + z[0], x[0] + y[0] + z[0])

    def maxYvalue(self, sbb, x1, y1, z1):

        x = x1 * sbb[0]
        y = y1 * sbb[1]
        z = z1 * sbb[2]

        return max(0, x[1], y[1], z[1], x[1] + y[1], x[1] + z[1], y[1] + z[1], x[1] + y[1] + z[1])

    def maxZvalue(self, sbb, x1, y1, z1):

        x = x1 * sbb[0]
        y = y1 * sbb[1]
        z = z1 * sbb[2]

        return max(0, x[2], y[2], z[2], x[2] + y[2], x[2] + z[2], y[2] + z[2], x[2] + y[2] + z[2])

    def minXvalue(self, sbb, x1, y1, z1):

        x = x1 * sbb[0]
        y = y1 * sbb[1]
        z = z1 * sbb[2]

        return min(0, x[0], y[0], z[0], x[0] + y[0], x[0] + z[0], y[0] + z[0], x[0] + y[0] + z[0])

    def minYvalue(self, sbb, x1, y1, z1):

        x = x1 * sbb[0]
        y = y1 * sbb[1]
        z = z1 * sbb[2]
        return min(0, x[1], y[1], z[1], x[1] + y[1], x[1] + z[1], y[1] + z[1], x[1] + y[1] + z[1])

    def minZvalue(self, sbb, x1, y1, z1):

        x = x1 * sbb[0]
        y = y1 * sbb[1]
        z = z1 * sbb[2]

        return min(0, x[2], y[2], z[2], x[2] + y[2], x[2] + z[2], y[2] + z[2], x[2] + y[2] + z[2])

    def vector_length(self, vector):
        return math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2) + math.pow(vector[2], 2))

    def minZvaluePoint(self, sbb, x1, y1, z1):

        x = x1 * sbb[0]
        y = y1 * sbb[1]
        z = z1 * sbb[2]

        minimum =  min(0, x[2], y[2], z[2], x[2] + y[2], x[2] + z[2], y[2] + z[2], x[2] + y[2] + z[2])
        values = [0, x[2], y[2], z[2], x[2] + y[2], x[2] + z[2], y[2] + z[2], x[2] + y[2] + z[2]]
        argminimum = values.index(minimum)

        return argminimum

    def upwardfacingVectors(self, x, y, z):
        if x[2] < 0:
            altx = [-x[0], -x[1], -x[2]]
        else:
            altx = [x[0], x[1], x[2]]
        if y[2] < 0:
            alty = [-y[0], -y[1], -y[2]]
        else:
            alty = [y[0], y[1], y[2]]
        if z[2] < 0:
            altz = [-z[0], -z[1], -z[2]]
        else:
            altz = [z[0], z[1], z[2]]

        return altx, alty, altz

    def chooserightAxis(self, index, x, y, z):
        if index == 0:
            return x
        if index == 1:
            return y
        if index == 2:
            return z

    def rotationfromWorldAxis(self, alpha, beta, gamma):
        r = R.from_euler('xyz', [alpha, beta, gamma], degrees=False)

        r_object = r.as_euler('XYZ', degrees=False)

        return r_object
