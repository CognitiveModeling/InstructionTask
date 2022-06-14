import random

try:
    import sim
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

import time
import shapes
import numpy as np
import json
import math


n_blocks = 3
n_actions = n_blocks + 2
max_cu = 3
max_cy = 2
max_s = 2
max_py = 2
max_co = 2

print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim
if clientID != -1:
    print('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        print('Number of objects in the scene: ', len(objs))
    else:
        print('Remote API function call returned with error code: ', res)

    time.sleep(2)

    # Now retrieve streaming data (i.e. in a non-blocking fashion):
    startTime = time.time()

    # define rgb values for colors
    black = (20 / 255, 20 / 255, 20 / 255)
    white = (248 / 255, 248 / 255, 248 / 255)
    red = (181 / 255, 37 / 255, 38 / 255)
    green = (0 / 255, 148 / 255, 60 / 255)
    yellow = (254 / 255, 182 / 255, 0 / 255)
    blue = (8 / 255, 145 / 255, 187 / 255)
    brown = (77 / 255, 38 / 255, 30 / 255)
    purple = (105 / 255, 52 / 255, 117 / 255)
    pink = (234 / 255, 152 / 255, 183 / 255)
    orange = (220 / 255, 65 / 255, 2 / 255)
    gray = (128 / 255, 128 / 255, 128 / 255)
    blueish = (28 / 255, 171 / 255, 174 / 255)
    greenish = (178 / 255, 179 / 255, 0)
    reddish = (132 / 255, 32 / 255, 57 / 255)
    yellowish = (234 / 255, 151 / 255, 0)
    brownish = (111 / 255, 88 / 255, 21 / 255)

    colors = [black, white, red, green, yellow, blue, brown, purple, pink, orange, gray, blueish, greenish, reddish,
              yellowish, brownish]

    even_list = list(range(9000, 10000))
    clean = False

    for j in even_list:
        print("trial: " + str(j))

        shapeslist = []
        rands = np.random.randint(5, size=n_blocks)
        cu = 0
        cy = 0
        s = 0
        py = 0
        co = 0

        m = list(range(6))
        left = random.choice(m)
        front = random.choice(m)

        # select shape objects
        for i in range(n_blocks):
            if rands[i] == 0:
                if cu < max_cu:
                    shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                    cu += 1
                else:
                    rands[i] = 1
            if rands[i] == 1:
                if cy < max_cy:
                    shapeslist.append(shapes.Shape(clientID, "Cylinder", cy))
                    cy += 1
                else:
                    rands[i] = 2
            if rands[i] == 2:
                if s < max_s:
                    shapeslist.append(shapes.Shape(clientID, "Sphere", s))
                    s += 1
                else:
                    rands[i] = 3
            if rands[i] == 3:
                if py < max_py:
                    shapeslist.append(shapes.Shape(clientID, "Pyramid", py))
                    py += 1
                else:
                    rands[i] = 4
            if rands[i] == 4:
                if co < max_co:
                    shapeslist.append(shapes.Shape(clientID, "Cone", co))
                    co += 1
                else:
                    shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                    cu += 1

        sample = []
        withoutAll = []

        for n in range(n_blocks):
            withoutn = shapeslist.copy()
            withoutn.remove(shapeslist[n])
            withoutAll.append(withoutn)

        reshape = []
        arrangement = []

        # create new order by shuffling
        old_order = list(range(n_blocks))
        np.random.shuffle(old_order)

        # make sure third object is selected at point 3 and 4 again
        add1 = old_order[2]
        add2 = old_order[2]

        order = [old_order[0], old_order[1], old_order[2], add1, add2]

        for shape in shapeslist:
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            # all sides of bounding box are the same length
            x = np.random.uniform(0.2, 1)
            y = x
            z = x

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            sample_input = []
            sample_target = []

            r, g, b = random.choice(colors)
            fx = 0
            fy = 0

            # set size and color of the shape
            shape.scale_shape(x, y, z)
            shape.set_color(r, g, b)
            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

        time.sleep(1)

        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

        time.sleep(1)
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        last_orientations = []
        last_positions = []

        for a in range(n_actions):
            print(a)
            timestep = []
            # get and save current state of all shapes
            for s in range(len(shapeslist)):
                shape = shapeslist[s]

                properties = [list(shape.get_position_adapted()), list(shape.get_orientation_type_simple()),
                              list(shape.get_color())]
                size = shape.get_bounding_box()[0]
                properties.append([size])

                if shapes.out_of_bounds(shape.get_raw_position()):
                    properties.append(0)
                else:
                    properties.append(1)
                properties.append(shape.shape_type_numbered)

                timestep.append(properties)

            fx = 0
            fy = 0
            orientation_type = [1, 0, 0]

            # randomize rotation depending on shape type
            rotation = [0, 0]
            if shapeslist[order[a]].shape_type_numbered == 2 or shapeslist[order[a]].shape_type_numbered == 1 or \
                    shapeslist[order[a]].shape_type_numbered == 4:
                rotation[0] = 1
                rotation[1] = 0
            else:
                rotation[0] = np.random.uniform(- 1, 1)
                rotation[1] = math.sin(math.acos(rotation[0]))
                if random.randint(0, 1) == 0:
                    rotation[1] = -rotation[1]

            # determine target position
            if j % 5 == 0:
                left_right = float(np.random.uniform(-0.8, 0.8))
                front_back = float(np.random.uniform(-0.8, 0.8))
                up = float(np.random.uniform(- 0.1, 1))
            else:
                left_right = float(np.random.uniform(left * 0.3 - 0.9, left * 0.3 - 0.8))
                front_back = float(np.random.uniform(front * 0.3 - 0.9, front * 0.3 - 0.8))
                if a == 0:
                    up = float(np.random.uniform(0.0, 0.1))
                elif a == 1:
                    up = float(np.random.uniform(0.2, 0.6))
                else:
                    up = float(np.random.uniform(0.3, 1.))
            print("x: " + str(left_right))
            print("y: " + str(front_back))
            print("z: " + str(up))

            shapeslist[order[a]].move_to(2, 2, [])

            time.sleep(1)
            # first, set the orientation
            shapeslist[order[a]].set_visual_orientation_simple([orientation_type[0], orientation_type[1],
                                                                orientation_type[2], rotation[0],
                                                                rotation[1]])

            orientation = list([orientation_type[0], orientation_type[1], orientation_type[2], rotation[0],
                                rotation[1]])
            # then set the position
            position = shapeslist[order[a]].set_position([left_right, front_back, up], withoutAll[order[a]])

            time.sleep(1)
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(2)

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            # save target rotation and position
            properties = [[int(order[a])], list(position), list(orientation)]

            timestep.append(list(properties))

            arrangement.append(list(timestep))

        timestep = []

        for s in range(len(shapeslist)):
            # get and save resulting state for all shapes
            shape = shapeslist[s]

            properties = []
            pos = shape.get_position_adapted()
            properties.append(list(pos))
            properties.append(list(shape.get_orientation_type_simple()))
            properties.append(list(shape.get_color()))
            properties.append([shape.get_bounding_box()[0]])
            if shapes.out_of_bounds(shape.get_raw_position()):
                properties.append(0)
            else:
                properties.append(1)
            properties.append(shape.shape_type_numbered)

            timestep.append(list(properties))

        arrangement.append(list(timestep))

        # save data
        with open("mixed_arrangements/arrangement" + str(j) + ".json", 'w') as f:
            json.dump(list(arrangement), f, indent=2)
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

        # reset shapes
        for i in range(len(shapeslist)):
            shapeslist[i].scale_shape(reshape[i][0], reshape[i][1], reshape[i][2])
            shapeslist[i].turn_original_way_up()
            shapeslist[i].set_position_eval([i * 4 - 4, 3.3, 1])

        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID, 'Hello CoppeliaSim!', sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive.
    # You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')
