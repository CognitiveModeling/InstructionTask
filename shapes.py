import torch

import sim
from scipy.spatial.transform import Rotation as R
import math


class Shape:  # properties of shapes in the CoppeliaSim simulation
    def __init__(self, client_ID, shape, number):
        self.type = shape.__class__.__name__
        self.shape_type = shape
        if shape != "Cuboid" and shape != "Cylinder" and shape != "Sphere" and shape != "Cone" and shape != "Pyramid":
            print("false name: " + shape)
            exit()
        elif number == 0:
            self.name = shape + "#"
        else:
            name = shape + "{}#"
            self.name = name.format(number - 1)
        if shape == "Cuboid":
            self.shape_type_numbered = 0
        elif shape == "Cylinder":
            self.shape_type_numbered = 1
        elif shape == "Sphere":
            self.shape_type_numbered = 2
        elif shape == "Cone":
            self.shape_type_numbered = 4
        elif shape == "Pyramid":
            self.shape_type_numbered = 3
        self.clientID = client_ID
        self.handle = self.get_handle()

    # "get" functions

    # marker of the object, for identification
    def get_handle(self):
        _, handle = sim.simxGetObjectHandle(self.clientID, self.name, sim.simx_opmode_blocking)
        return handle

    # returns the color of the object as rgb vector
    def get_color(self):
        object_handle = self.handle
        return_code, _, color, _, _ = sim.simxCallScriptFunction(self.clientID, "ForScript",
                                                                 sim.sim_scripttype_childscript,
                                                                 "getShapeColor_function",
                                                                 [object_handle], [], [],
                                                                 bytearray(),
                                                                 sim.simx_opmode_blocking)
        if return_code == sim.simx_return_ok:
            return color
        else:
            return self.get_color()

    # returns absolute position of the object in world coordinates (x, y, z)
    def get_raw_position(self):
        return_code, position = sim.simxGetObjectPosition(self.clientID, self.handle, -1, sim.simx_opmode_blocking)
        if return_code == sim.simx_return_ok:
            return position
        else:
            return return_code

    # corrects the absolute world coordinates to only within-field coordinates
    def get_position_clean(self):
        position = self.get_raw_position()

        if out_of_bounds(position):
            return [0, 0, 0]
        else:
            return position

    # corrects absolute position's z value to representing lowest point of object instead of center
    def get_position_adapted(self):
        position = self.get_raw_position()

        if out_of_bounds(position):
            return [-1, -1, -1]
        else:
            bb = self.get_bounding_box_world()[2]
            position[2] = position[2] - 0.5 * bb
            if same_as(position[2], 0):
                position[2] = 0
            return position

    # raw rotation of the object, in radian values
    def get_radian_orientation(self):
        _, orientation = sim.simxGetObjectOrientation(self.clientID, self.handle, -1,
                                                      sim.simx_opmode_blocking)
        return orientation

    # converts radian values of object rotation to sin and cos
    def get_orientation(self):
        orientation = [-1, -1, -1, -1, -1, -1]
        if not out_of_bounds(self.get_raw_position()):
            _, rad_orientation = sim.simxGetObjectOrientation(self.clientID, self.handle, -1,
                                                              sim.simx_opmode_blocking)

            orientation[0] = math.sin(rad_orientation[0])
            orientation[1] = math.cos(rad_orientation[0])
            orientation[2] = math.sin(rad_orientation[1])
            orientation[3] = math.cos(rad_orientation[1])
            orientation[4] = math.sin(rad_orientation[2])
            orientation[5] = math.cos(rad_orientation[2])
        return orientation

    # simplifies rotation given the object type
    def get_orientation_type_simple(self):
        o = self.get_radian_orientation()

        if out_of_bounds(self.get_raw_position()):
            return [-1, -1, -1, -1, -1]
        if self.shape_type_numbered == 2:
            return [1, 0, 0, 1, 0]

        or_type = [0, 0, 0, 0, 0]

        ex = math.degrees(o[0])
        ey = math.degrees(o[1])
        ez = math.degrees(o[2])

        r = R.from_euler('XYZ', [ex, ey, ez], degrees=True)

        x = get_x_vector(r)
        y = get_y_vector(r)
        z = get_z_vector(r)

        axes = [x, y, z]

        flat, axis = is_object_flat_on_ground(x, y, z)
        if self.shape_type_numbered == 1:
            if axis != 2:
                flat = False

        if flat:
            or_type[0] = 1

            axes.pop(axis)

            if self.shape_type_numbered == 1 or self.shape_type_numbered == 2 or self.shape_type_numbered == 4:
                or_type[3] = 1
                or_type[4] = 0
            else:
                if abs(axes[0][1]) < abs(axes[1][1]):
                    ref_axis = axes[0]
                else:
                    ref_axis = axes[1]

                or_type[3] = ref_axis[1]
                if ref_axis[0] < 0:
                    or_type[3] = -or_type[3]

                or_type[3], or_type[4] = encode_orientation(or_type[3])

        else:
            edge, axis = is_object_on_edge(x, y, z)

            if edge:
                ref_axis = axes[axis]
                or_type[1] = 1
                axes.pop(axis)

                or_type[3] = ref_axis[1]
                if ref_axis[0] < 0:
                    or_type[3] = -or_type[3]

                or_type[3], or_type[4] = encode_orientation(or_type[3])

            else:
                or_type[2] = 1

        return or_type

    # bounding box of the object, in relation to object coordinate system
    def get_bounding_box(self):
        bounding_box = [0, 0, 0]

        _, bounding_box[0] = sim.simxGetObjectFloatParameter(self.clientID, self.handle,
                                                             sim.sim_objfloatparam_objbbox_max_x,
                                                             sim.simx_opmode_blocking)
        bounding_box[0] *= 2
        _, bounding_box[1] = sim.simxGetObjectFloatParameter(self.clientID, self.handle,
                                                             sim.sim_objfloatparam_objbbox_max_y,
                                                             sim.simx_opmode_blocking)
        bounding_box[1] *= 2
        _, bounding_box[2] = sim.simxGetObjectFloatParameter(self.clientID, self.handle,
                                                             sim.sim_objfloatparam_objbbox_max_z,
                                                             sim.simx_opmode_blocking)
        bounding_box[2] *= 2
        return bounding_box

    # bounding box of the object, in relation to world coordinates
    def get_bounding_box_world(self):
        sbb = self.get_bounding_box()

        bounding_box_world = [0, 0, 0]

        x, y, z = self.get_vectors()

        max_xvalue = max_x_value(sbb, x, y, z)
        max_yvalue = max_y_value(sbb, x, y, z)
        max_zvalue = max_z_value(sbb, x, y, z)
        min_xvalue = min_x_value(sbb, x, y, z)
        min_yvalue = min_y_value(sbb, x, y, z)
        min_zvalue = min_z_value(sbb, x, y, z)

        bounding_box_world[0] = abs(max_xvalue - min_xvalue)
        bounding_box_world[1] = abs(max_yvalue - min_yvalue)
        bounding_box_world[2] = abs(max_zvalue - min_zvalue)

        return bounding_box_world

    # returns all object bounding box edge points in world coordinates
    def get_edge_points(self, position):
        x, y, z = self.get_vectors()
        bb = self.get_bounding_box()

        x1 = vector_multiplication(x, bb[0] * 0.5)
        x2 = vector_negation(x1)
        y1 = vector_multiplication(y, bb[1] * 0.5)
        y2 = vector_negation(y1)
        z1 = vector_multiplication(z, bb[2] * 0.5)
        z2 = vector_negation(z1)

        p1 = vector_addition(vector_addition(vector_addition(x1, y1), z1), position)
        p2 = vector_addition(vector_addition(vector_addition(x1, y1), z2), position)
        p3 = vector_addition(vector_addition(vector_addition(x1, y2), z1), position)
        p4 = vector_addition(vector_addition(vector_addition(x1, y2), z2), position)
        p5 = vector_addition(vector_addition(vector_addition(x2, y1), z1), position)
        p6 = vector_addition(vector_addition(vector_addition(x2, y1), z2), position)
        p7 = vector_addition(vector_addition(vector_addition(x2, y2), z1), position)
        p8 = vector_addition(vector_addition(vector_addition(x2, y2), z2), position)

        return p1, p2, p3, p4, p5, p6, p7, p8

    # returns number that represents the shape type
    def get_type(self):
        return self.shape_type_numbered

    # returns vectors of object coordinate axes in world coordinates
    def get_vectors(self):
        o = self.get_radian_orientation()

        ex = math.degrees(o[0])
        ey = math.degrees(o[1])
        ez = math.degrees(o[2])

        r = R.from_euler('XYZ', [ex, ey, ez], degrees=True)

        x = get_x_vector(r)
        y = get_y_vector(r)
        z = get_z_vector(r)

        return x, y, z

    # "set" functions

    # sets object color in rgb values
    def set_color(self, r, g, b):
        object_handle = self.handle
        sim.simxCallScriptFunction(self.clientID, "ForScript", sim.sim_scripttype_childscript,
                                   "setShapeColor_function", [object_handle], [r, g, b], [],
                                   bytearray(), sim.simx_opmode_blocking)

    # adapts object x, y, z length by multiplying with respective value
    def scale_shape(self, x, y, z):
        object_handle = self.handle
        sim.simxCallScriptFunction(self.clientID, "ForScript", sim.sim_scripttype_childscript,
                                   "setObjectShape_function", [object_handle], [x, y, z],
                                   [], bytearray(), sim.simx_opmode_blocking)

    # sets object position, sets outside of field when collision with ground or other object
    def set_position(self, position, shape_list):
        bb = self.get_bounding_box_world()
        collision, shape = self.collision(position, shape_list)
        below = below_ground(position, bb)
        out = out_of_bounds(position)
        if torch.is_tensor(position):
            res_position = position.clone()
        else:
            res_position = position.copy()
        if out:
            print("out of bounds")
        elif collision:
            print("collision")
            res_position = [-3.5, 3.5, 1]
        elif below:
            print("below ground")
            res_position = [-3.5, 3.5, 1]

        sim.simxSetObjectPosition(self.clientID, self.handle, -1, res_position, sim.simx_opmode_blocking)

        return position

    # sets position without checking for collisions
    def set_position_eval(self, position):
        sim.simxSetObjectPosition(self.clientID, self.handle, -1, position, sim.simx_opmode_blocking)

    # rotates object, takes sin/cos rotation of length 6
    def set_orientation(self, orientation):
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
            if orientation[2 * i] >= 0 >= orientation[2 * i + 1]:
                rad_or[i] = a_cos[i]
            elif orientation[2 * i] <= 0 and orientation[2 * i + 1] <= 0:
                rad_or[i] = - a_cos[i]
            elif orientation[2 * i] <= 0 <= orientation[2 * i + 1]:
                rad_or[i] = a_sin[i]
            elif orientation[2 * i] >= 0 and orientation[2 * i + 1] >= 0:
                rad_or[i] = a_sin[i]
            else:
                print("Error: no orientation")

        sim.simxSetObjectOrientation(self.clientID, self.handle, -1, rad_or, sim.simx_opmode_blocking)

    # takes care of old version, simplifies to simple plane based rotation
    def set_visual_orientation_simple(self, orig_orientation):
        orientation = [0, 0, 0, 0]
        orientation[0] = orig_orientation[0]
        orientation[1] = orig_orientation[1]
        orientation[2] = orig_orientation[2]
        orientation[3] = decode_orientation(orig_orientation[3], orig_orientation[4])

        self.set_plane_orientation_simple(orientation)

    def set_plane_orientation_simple(self, orientation):
        rotation = math.asin(orientation[3])

        new_or = rotate_plane_orientation_simple(rotation)

        self.set_radian_orientation(new_or)

    # sets rotation using radian vector of length 3
    def set_radian_orientation(self, orientation):
        sim.simxSetObjectOrientation(self.clientID, self.handle, -1, orientation, sim.simx_opmode_blocking)

    # sets object on the ground at position x, y
    def move_to(self, x, y, shape_list):
        position = [x, y, self.get_bounding_box_world()[2] * 0.5]
        position = self.set_position(position, shape_list)

        return position

    # rotates object back to align z value with world coordinate system
    def turn_original_way_up(self):
        self.set_orientation([0, 1, 0, 1, self.get_orientation()[4], self.get_orientation()[5]])

    # helpers

    # specifies if a hypothetical position would make object collide with other object and if so,
    # returns collision object
    def collision(self, position, shape_list):
        p1, p2, p3, p4, p5, p6, p7, p8 = self.get_edge_points(position)
        points = [p1, p2, p3, p4, p5, p6, p7, p8]

        plane1 = get_plane(p1, p5)
        plane2 = get_plane(p1, p2)
        plane3 = get_plane(p1, p3)

        plane11 = get_plane(p8, p4)
        plane21 = get_plane(p8, p7)
        plane31 = get_plane(p8, p6)

        planes = [[plane1, plane2, plane3, plane11, plane21, plane31], [[p1, p2, p3], [p1, p3, p5], [p1, p2, p5],
                                                                        [p8, p7, p6], [p8, p4, p6], [p8, p4, p7]]]
        line1 = get_line(p1, p2)
        line2 = get_line(p1, p5)
        line3 = get_line(p5, p6)
        line4 = get_line(p2, p6)
        line5 = get_line(p3, p4)
        line6 = get_line(p3, p7)
        line7 = get_line(p7, p8)
        line8 = get_line(p4, p8)
        line9 = get_line(p5, p7)
        line10 = get_line(p1, p6)
        line11 = get_line(p2, p4)
        line12 = get_line(p6, p8)

        lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12]
        for obj in shape_list:
            op1, op2, op3, op4, op5, op6, op7, op8 = obj.get_edge_points(obj.get_raw_position())
            opoints = [op1, op2, op3, op4, op5, op6, op7, op8]

            oplane1 = get_plane(op1, op5)
            oplane2 = get_plane(op1, op2)
            oplane3 = get_plane(op1, op3)

            oplane11 = get_plane(op8, op4)
            oplane21 = get_plane(op8, op7)
            oplane31 = get_plane(op8, op6)

            oplanes = [[oplane1, oplane2, oplane3, oplane11, oplane21, oplane31], [[op1, op2, op3], [op1, op3, op5],
                                                                                   [op1, op2, op5], [op8, op7, op6],
                                                                                   [op8, op4, op6], [op8, op4, op7]]]

            oline1 = get_line(op1, op2)
            oline2 = get_line(op1, op5)
            oline3 = get_line(op5, op6)
            oline4 = get_line(op2, op6)
            oline5 = get_line(op3, op4)
            oline6 = get_line(op3, op7)
            oline7 = get_line(op7, op8)
            oline8 = get_line(op4, op8)
            oline9 = get_line(op5, op7)
            oline10 = get_line(op1, op6)
            oline11 = get_line(op2, op4)
            oline12 = get_line(op6, op8)

            olines = [oline1, oline2, oline3, oline4, oline5, oline6, oline7, oline8, oline9, oline10, oline11, oline12]

            for line in olines:
                for i in range(6):
                    parallel, intersection, r = intersect(planes[0][i], line)
                    if not parallel:
                        # test if intersection lies between the edge points
                        if 0 <= r <= 1:
                            # test if intersection lies in rectangle of plane
                            if in_rectangle(intersection, planes[1][i]):
                                return True, obj

            for line in lines:
                for i in range(6):
                    parallel, intersection, r = intersect(oplanes[0][i], line)
                    if not parallel:
                        # test if intersection lies between the edge points
                        if 0 <= r <= 1:
                            # test if intersection lies in rectangle of plane
                            if in_rectangle(intersection, oplanes[1][i]):
                                return True, obj
            d = 1
            for point in points:
                for i in range(6):
                    d = get_distance_from_plane(oplanes[0][i], point)
                    if d < 0:
                        break
                if d < 0:
                    break
            if d >= 0:
                return True, obj

            d = 1
            for point in opoints:
                for i in range(6):
                    d = get_distance_from_plane(planes[0][i], point)
                    if d < 0:
                        break
                if d < 0:
                    break
            if d >= 0:
                return True, obj

        return False, 0


# adds margin of slack to equal function
def same_as(v1, v2):
    return v2 - 0.05 < v1 < v2 + 0.05


# distance between two points
def point_distance(p1, p2):
    return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))


# is value a the largest of the three?
def is_largest(a, b, c):
    return a > b and a > c


# is value a the smallest of the three?
def is_smallest(a, b, c):
    return a < b and a < c


# given bounding box and position, would there be a collision with the ground?
def below_ground(position, bb):
    z = position[2]
    bbz = bb[2] * 0.5

    return z - bbz < 0


# is the position outside the field?
def out_of_bounds(position):
    return not (2.5 > position[0] > -2.5 and 2.5 > position[1] > -2.5)


# get vector of x axis given rotation r
def get_x_vector(r):
    x_vector = r.apply([1, 0, 0])
    return x_vector


# get vector of y axis given rotation r
def get_y_vector(r):
    y_vector = r.apply([0, 1, 0])

    return y_vector


# get vector of z axis given rotation r
def get_z_vector(r):
    z_vector = r.apply([0, 0, 1])

    return z_vector


# returns normal vector form of plane
def get_plane(p1, p2):
    n = vector_subtraction(p2, p1)
    no = vector_division(n, vector_length(n))

    return [no, p1]


# rotates a plane - simplified
def rotate_plane_orientation_simple(rotation):
    new_or = [0, 0, 0]
    new_or[2] = rotation

    return new_or


# is point s in the rectangle defined by points?
def in_rectangle(s, points):
    a = vector_subtraction(points[1], points[0])
    b = vector_subtraction(points[2], points[0])
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


# scalar product of two vectors
def scalar_product(p, n):
    return p[0] * n[0] + p[1] * n[1] + p[2] * n[2]


# length of vector p
def vector_length(p):
    return math.sqrt(math.pow(p[0], 2) + math.pow(p[1], 2) + math.pow(p[2], 2))


# divides vector p by value n
def vector_division(p, n):
    return [p[0] / n, p[1] / n, p[2] / n]


# negates vector p
def vector_negation(p):
    return [-p[0], -p[1], -p[2]]


# vector p1 - vector p2
def vector_subtraction(p1, p2):
    return [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]


# vector p1 + vector p2
def vector_addition(p1, p2):
    return [p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]]


# vector p * scalar
def vector_multiplication(p, scalar):
    return [p[0] * scalar, p[1] * scalar, p[2] * scalar]


# returns distance of point p from plane
def get_distance_from_plane(plane, p):
    return scalar_product(plane[0], vector_subtraction(p, plane[1]))


# checks if two lines are (vaguely) parallel
def lines_parallel(line1, line2):
    scalar = scalar_product(line1[1], line2[1])
    len1 = vector_length(line1[1])
    len2 = vector_length(line2[1])
    product = len1 * len2

    return product - 0.005 < scalar < product + 0.005


# cross product of vectors v1 and v2
def cross_product(v1, v2):
    return [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]]


# returns line that goes through both p1 and p2
def get_line(p1, p2):
    return [p1, vector_subtraction(p2, p1)]


# returns the distance of a point from a line
def distance_point_line(point, line):
    help_vector = vector_subtraction(point, line[0])
    cp = cross_product(help_vector, line[1])
    if vector_length(line[1]) == 0:
        print("length of line 0")
        print(line)
        exit()
    d = vector_length(cp) / vector_length(line[1])

    return d


# do plane and line not intersect? returns true if parallel; returns intersection point if false
def intersect(plane, line):
    if scalar_product(plane[0], line[1]) == 0:
        return True, [], 0

    r = (scalar_product(plane[0], plane[1]) - scalar_product(plane[0], line[0])) / scalar_product(
        plane[0], line[1])
    p = vector_addition(line[0], vector_multiplication(line[1], r))
    return False, p, r


# checks if an object is flat on the ground given its coordinate axes vectors in world coordinates
def is_object_flat_on_ground(x, y, z):
    if 0.99 < x[2] < 1.01 or -0.99 > x[2] > -1.01:
        return True, 0
    elif 0.99 < y[2] < 1.01 or -0.99 > y[2] > -1.01:
        return True, 1
    elif 0.99 < z[2] < 1.01 or -0.99 > z[2] > -1.01:
        return True, 2
    else:
        return False, -1


# checks if an object is standing on an edge given its coordinate axes vectors in world coordinates
def is_object_on_edge(x, y, z):
    if -0.05 < x[2] < 0.05:
        return True, 0
    elif -0.05 < y[2] < 0.05:
        return True, 1
    elif -0.05 < z[2] < 0.05:
        return True, 2
    else:
        return False, -1


# returns x value of point furthest in x direction given bounding box and coordinate axes vectors of an object
def max_x_value(sbb, x1, y1, z1):
    x = x1 * sbb[0]
    y = y1 * sbb[1]
    z = z1 * sbb[2]

    return max(0, x[0], y[0], z[0], x[0] + y[0], x[0] + z[0], y[0] + z[0], x[0] + y[0] + z[0])


# returns y value of point furthest in y direction given bounding box and coordinate axes vectors of an object
def max_y_value(sbb, x1, y1, z1):
    x = x1 * sbb[0]
    y = y1 * sbb[1]
    z = z1 * sbb[2]

    return max(0, x[1], y[1], z[1], x[1] + y[1], x[1] + z[1], y[1] + z[1], x[1] + y[1] + z[1])


# returns z value of highest point given bounding box and coordinate axes vectors of an object
def max_z_value(sbb, x1, y1, z1):
    x = x1 * sbb[0]
    y = y1 * sbb[1]
    z = z1 * sbb[2]

    return max(0, x[2], y[2], z[2], x[2] + y[2], x[2] + z[2], y[2] + z[2], x[2] + y[2] + z[2])


# returns x value of point furthest in negative x direction given bounding box and coordinate axes vectors of an object
def min_x_value(sbb, x1, y1, z1):
    x = x1 * sbb[0]
    y = y1 * sbb[1]
    z = z1 * sbb[2]

    return min(0, x[0], y[0], z[0], x[0] + y[0], x[0] + z[0], y[0] + z[0], x[0] + y[0] + z[0])


# returns y value of point furthest in negative y direction given bounding box and coordinate axes vectors of an object
def min_y_value(sbb, x1, y1, z1):
    x = x1 * sbb[0]
    y = y1 * sbb[1]
    z = z1 * sbb[2]
    return min(0, x[1], y[1], z[1], x[1] + y[1], x[1] + z[1], y[1] + z[1], x[1] + y[1] + z[1])


# returns z value of lowest point given bounding box and coordinate axes vectors of an object
def min_z_value(sbb, x1, y1, z1):
    x = x1 * sbb[0]
    y = y1 * sbb[1]
    z = z1 * sbb[2]

    return min(0, x[2], y[2], z[2], x[2] + y[2], x[2] + z[2], y[2] + z[2], x[2] + y[2] + z[2])


# computes cos and sin of radian value, correcting for negative values
def encode_orientation(value):
    if value > 0:
        pi_help = math.pi / math.sqrt(.5) * value
    else:
        pi_help = math.pi + math.pi / math.sqrt(.5) * (math.sqrt(.5) + value)

    cos_value = math.cos(pi_help)
    sin_value = math.sin(pi_help)

    return cos_value, sin_value


# computes radian value out of sin and cos values
def decode_orientation(cos_value, sin_value):
    asin_value = math.asin(sin_value)
    neg_asin = math.asin(-sin_value)
    acos_value = math.acos(cos_value)

    if sin_value >= 0:
        if cos_value >= 0:
            mean = (acos_value + asin_value) / 2
        else:
            mean = (acos_value + math.pi - asin_value) / 2
    else:
        if cos_value < 0:
            mean = - (acos_value + math.pi - neg_asin) / 2
        else:
            mean = - (acos_value + neg_asin) / 2

    return mean * math.sqrt(.5) / math.pi
