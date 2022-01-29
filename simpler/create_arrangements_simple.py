import random

try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import shapes
import numpy as np
import json
import mathown
import math


def same_position(po1, po2):
    return po1[0] - 0.01 <= po2[0] <= po1[0] + 0.01 and po1[1] - 0.01 <= po2[1] <= po1[1] + 0.01 and po1[2] - 0.2 <= \
           po2[2] <= po1[2] + 0.2


def same_position_other_block(po1, po2):
    return po1[0] - 0.2 <= po2[0] <= po1[0] + 0.2 and po1[1] - 0.2 <= po2[1] <= po1[1] + 0.2 and po1[2] - 0.1 <= \
           po2[2] <= po1[2] + 0.1


def same_orientation(po1, po2):
    if po1[0] == po2[0] and po1[1] == po2[1] and po1[2] == po2[2]:
        return po1[3] - 0.1 <= po2[3] <= po1[3] + 0.1 and po1[4] - 0.1 <= po2[4] <= po1[4] + 0.1


def out_of_bounds(position):
    return not (2.5 > position[0] > -2.5 and 2.5 > position[1] > -2.5)


def correct_position(position):
    if position[2] != -1 and position[2] < 0:
        position[2] = 0
    elif position[2] != -1 and position[2] != 0:
        position[2] = position[2] + 0.01
    return position

n_blocks = 3
n_actions = n_blocks + 2
max_cu = 5
max_cy = 0
max_s = 0

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res,objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print ('Number of objects in the scene: ',len(objs))
    else:
        print('Remote API function call returned with error code: ',res)

    time.sleep(2)

    # Now retrieve streaming data (i.e. in a non-blocking fashion):
    startTime=time.time()

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
    blueish = (28/255, 171/255, 174/255)
    greenish = (178/255, 179/255, 0)
    reddish = (132/255, 32/255, 57/255)
    yellowish = (234/255, 151/255, 0)
    brownish = (111/255, 88/255, 21/255)

    colors = [black, white, red, green, yellow, blue, brown, purple, pink, orange, gray, blueish, greenish, reddish,
              yellowish, brownish]

    cuboid1 = shapes.Shape(clientID, "Cuboid", 0)
    cuboid2 = shapes.Shape(clientID, "Cuboid", 1)
    cuboid3 = shapes.Shape(clientID, "Cuboid", 2)
    cuboid4 = shapes.Shape(clientID, "Cuboid", 3)
    cuboid5 = shapes.Shape(clientID, "Cuboid", 4)
    sphere1 = shapes.Shape(clientID, "Sphere", 0)
    sphere2 = shapes.Shape(clientID, "Sphere", 1)
    #sphere3 = shapes.Shape(clientID, "Sphere", 2)
    #sphere4 = shapes.Shape(clientID, "Sphere", 3)
    cylinder1 = shapes.Shape(clientID, "Cylinder", 0)
    cylinder2 = shapes.Shape(clientID, "Cylinder", 1)
    cylinder3 = shapes.Shape(clientID, "Cylinder", 2)
    cylinder4 = shapes.Shape(clientID, "Cylinder", 3)
    cylinder5 = shapes.Shape(clientID, "Cylinder", 4)

    allshapes = [cuboid1, cuboid3, cuboid4, cuboid2, cuboid5, cylinder3, cylinder2, cylinder4, cylinder1, cylinder5,
                 sphere2, sphere1]

    even_list = list(range(4590, 10000))
    clean = False

    for j in even_list:
        print("trial: " + str(j))

        shapeslist = []
        rands = np.random.randint(3, size=n_blocks)
        actions = list(range(5))
        #actions = [0]
        actionChoices = random.choices(actions, k=n_actions)
        cu = 0
        cy = 0
        s = 0

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

        old_order = list(range(n_blocks))
        np.random.shuffle(old_order)

        first_block = old_order[0]
        old_order.remove(first_block)

        add1 = np.random.choice(old_order)
        add2 = np.random.choice(old_order)

        order = list([old_order[0], old_order[1], add1, add2])
        np.random.shuffle(order)

        #print(first_block)
        #print(order)

        for shape in shapeslist:

            #shape.moveTo(shape.getPosition()[0], 0)
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

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

            shape.scaleShape(x, y, z)

            shape.setColor(r, g, b)
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

            if a > 1:
                new = shapeslist[order[a-2]].get_relative_position_old(shapeslist[first_block],
                                                                       shapeslist[order[a-2]].getPosition(),
                                                                       first=False)
                if not same_position(position, new):
                    print(new)
                    print(position)
                    shapeslist[order[a-2]].setPosition([-3.5, 3.5, 1], withoutAll[order[a-2]])
                    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
                    time.sleep(1)
                    sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            for s in range(len(shapeslist)):
                shape = shapeslist[s]

                if a == 0:
                    properties = []
                    properties.append(shape.getType())
                    properties.append(list(shape.getColor()))
                    properties.append([shape.getBoundingBox()[0]])
                    properties.append(list(shape.get_relative_position_simple(shapeslist[first_block],
                                                                              shape.getPosition(), True,
                                                                              withoutAll[s])))
                    properties.append(list(shape.getOrientationType_simple()))
                    properties.append(list(shape.getDistances(withoutAll[s])))

                    timestep.append(list(properties))
                    last_positions.append(shape.get_relative_position_old(shapeslist[first_block],
                                                                              shape.getPosition(), True))
                    last_orientations.append(shape.getOrientationType_simple())
                else:
                    properties = []
                    properties.append(shape.getType())
                    properties.append(list(shape.getColor()))
                    properties.append([shape.getBoundingBox()[0]])
                    new_pos = shape.get_relative_position_old(shapeslist[first_block], shape.getPosition(), first=False)
                    if a == 1:
                        print("shape:" + str(s))
                        print("first")
                        print(last_positions)
                        #print(last_orientations)
                        if s == first_block:
                            properties.append(
                                list(shape.get_relative_position_simple(shapeslist[first_block], shape.getPosition(),
                                                                       False, withoutAll[s])))
                        elif not same_position_other_block(new_pos, last_positions[s]) or not \
                                same_orientation(shape.getOrientationType_simple(), last_orientations[s]):
                            shape.setVisualOrientation_simple(last_orientations[s])
                            shape.set_relative_position_simple(last_positions[s], shapeslist[first_block],
                                                               withoutAll[s])
                            properties.append(
                                list(shape.get_relative_position_simple(shapeslist[first_block], shape.getPosition(),
                                                                        False, withoutAll[s])))
                    elif s == order[a-2]:
                        print("shape:" + str(s))
                        print("moved block fine")
                        print(last_positions)
                        properties.append(
                            list(shape.get_relative_position_simple(shapeslist[first_block], shape.getPosition(),
                                                                    False, withoutAll[s])))

                    elif s != first_block and (
                                not same_position_other_block(new_pos, last_positions[s]) or not same_orientation(
                            shape.getOrientationType_simple(), last_orientations[s])):
                        print("shape:" + str(s))
                        print("non moved block wrong")
                        print(last_positions)
                        print(new_pos)
                        #print(last_orientations)
                        shape.setVisualOrientation_simple(last_orientations[s])
                        shape.set_relative_position_simple(last_positions[s], shapeslist[first_block],
                                                           withoutAll[s])
                        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
                        time.sleep(1)
                        sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
                        properties.append(
                            list(shape.get_relative_position_simple(shapeslist[first_block], shape.getPosition(),
                                                                    False, withoutAll[s])))
                    else:
                        print("shape:" + str(s))
                        print("non moved block fine")
                        print(last_positions)
                        #print(last_orientations)
                        properties.append(
                            list(shape.get_relative_position_simple(shapeslist[first_block], shape.getPosition(),
                                                                    False, withoutAll[s])))
                    properties.append(list(shape.getOrientationType_simple()))
                    properties.append(list(shape.getDistances(withoutAll[s])))

                    timestep.append(list(properties))
                    last_positions[s] = correct_position(shape.get_relative_position_old(shapeslist[first_block],
                                                                                         shape.getPosition(), False))
                    last_orientations[s] = shape.getOrientationType_simple()

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(2)
            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            fx = 0
            fy = 0

            orientation_type = [0, 0, 0]

            type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
            type_choice = np.random.choice(type)

            orientation_type[type_choice] = 1

            facing_choices = [0, 0, 0]
            facing = [1, 0]

            if type_choice == 0:
                facing_choices[0] = np.random.uniform(- 1, 1)
                facing_choices[1] = math.sin(math.acos(facing_choices[0]))
                if random.randint(0, 1) == 0:
                    facing_choices[1] = -facing_choices[1]
            elif type_choice == 1:
                facing_choices[0] = np.random.uniform(- 1, 1)
                facing_choices[1] = math.sin(math.acos(facing_choices[0]))
                if random.randint(0, 1) == 0:
                    facing_choices[1] = -facing_choices[1]
                facing = [0, 1]
                facing_choice = random.choice(facing)
                facing_choices[2] = facing_choice


            if a == 0:
                sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
                shapeslist[first_block].moveTo(2, 2, [])
                shapeslist[first_block].setVisualOrientation_simple([orientation_type[0], orientation_type[1],
                                                                     orientation_type[2], facing_choices[0],
                                                                     facing_choices[1], facing_choices[2]])
                time.sleep(1)
                shapeslist[first_block].moveTo(fx, fy, [])
                orientation = [orientation_type[0], orientation_type[1], orientation_type[2], facing_choices[0],
                               facing_choices[1], facing_choices[2]]
                #print(list([o1, o2, o3, o4, o5, o6]))
                #orientation = shapeslist[first_block].getOrientationType()
                #print(orientation)
                position = [0, 0, 0]

                time.sleep(1)
                sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
                time.sleep(3)

                sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
                # getting final orientation as target orientation
                #orientation = shapeslist[first_block].getOrientationType_simple()
                #abs_pos = shapeslist[first_block].getPosition()
                #position = shapeslist[first_block].getRelativePosition(shapeslist[first_block], abs_pos, False)

                properties = []
                properties.append([int(first_block)])
                properties.append(list(position))
                properties.append(list(orientation))

                timestep.append(list(properties))

                arrangement.append(list(timestep))

            else:
                leftright = np.random.uniform(-0.5, 0.5)
                frontback = np.random.uniform(-0.5, 0.5)
                up = np.random.uniform(- 0.3, 0.2)
                print("up: " + str(up))
                shapeslist[order[a-1]].moveTo(2, 2, [])
                shapeslist[order[a-1]].setVisualOrientation_simple([orientation_type[0], orientation_type[1],
                                                                    orientation_type[2], facing_choices[0],
                                                                    facing_choices[1], facing_choices[2]])
                #sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
                orientation = list([orientation_type[0], orientation_type[1], orientation_type[2], facing_choices[0],
                               facing_choices[1], facing_choices[2]])
                #orientation = list([o1, o2, o3, o4, o5, o6])
                #orientation = shapeslist[order[a-1]].getOrientationType()
                action = j % 5
                #position = shapeslist[order[a - 1]].perform_random_simple_action(shapeslist[first_block],
                #                                                                 action, leftright, frontback, up,
                #
                #                                                                 withoutAll[order[a - 1]])
                if not clean:
                    position = shapeslist[order[a-1]].perform_random_simple_action(shapeslist[first_block], actionChoices[a],
                                                                              leftright, frontback, up,
                                                                               withoutAll[order[a-1]])
                else:
                    position = shapeslist[order[a - 1]].perform_random_simple_action_alt(shapeslist[first_block],
                                                                                     actionChoices[a],
                                                                                     leftright, frontback, up,
                                                                                     withoutAll[order[a - 1]])

                #print(shapeslist[order[a-1]])
                #print(withoutAll[order[a-1]])

                time.sleep(1)
                sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
                time.sleep(3)

                sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
                #getting final orientation as target orientation
                #orientation = shapeslist[order[a-1]].getOrientationType_simple()
                #abs_pos = shapeslist[order[a-1]].getPosition()
                #position = shapeslist[order[a-1]].getRelativePosition(shapeslist[first_block], abs_pos, False)

                properties = []
                properties.append([int(order[a-1])])
                properties.append(list(position))
                properties.append(list(orientation))

                timestep.append(list(properties))

                arrangement.append(list(timestep))


        timestep = []
        new = shapeslist[order[-1]].get_relative_position_old(shapeslist[first_block],
                                                              shapeslist[order[-1]].getPosition(), first=False)
        if not same_position(position, new):
            print(new)
            print(position)
            shapeslist[order[-1]].setPosition([-3.5, 3.5, 1], withoutAll[order[-1]])
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(1)
            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

        for s in range(len(shapeslist)):
            shape = shapeslist[s]

            properties = []
            properties.append(shape.getType())
            properties.append(list(shape.getColor()))
            properties.append([shape.getBoundingBox()[0]])

            new_pos = shape.get_relative_position_old(shapeslist[first_block], shape.getPosition(), first=False)
            if s == order[-1]:
                print("shape:" + str(s))
                print("moved block fine")
                print(last_positions)
                properties.append(list(shape.get_relative_position_simple(shapeslist[first_block],
                                                                          shape.getPosition(), False,
                                                                          withoutAll[s])))
            elif s != first_block and (not same_position_other_block(new_pos, last_positions[s]) or not same_orientation(
                        shape.getOrientationType_simple(), last_orientations[s])):
                print("shape:" + str(s))
                print("non moved block wrong")
                shape.setVisualOrientation_simple(last_orientations[s])
                shape.set_relative_position_simple(last_positions[s], shapeslist[first_block],
                                                   withoutAll[s])
                sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
                time.sleep(1)
                sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
            else:
                print("shape:" + str(s))
                print("non moved block fine")
                properties.append(list(shape.get_relative_position_simple(shapeslist[first_block], shape.getPosition(),
                                                                          False, withoutAll[s])))
            properties.append(list(shape.getOrientationType_simple()))
            properties.append(list(shape.getDistances(withoutAll[s])))

            #print([shape.getBoundingBox()[0]])
            #print(shape.getDistances(withoutAll[s]))

            timestep.append(list(properties))

        #input()

        arrangement.append(list(timestep))

        with open("arrangements/arrangement" + str(j) + ".json", 'w') as f:
            json.dump(list(arrangement), f, indent=2)
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

        #input()

        for i in range(len(shapeslist)):
            #shapeslist[i].setPosition([i-4, 3, shapeslist[i].getPosition()[2]])
            shapeslist[i].scaleShape(reshape[i][0], reshape[i][1], reshape[i][2])
            shapeslist[i].turnOriginalWayUp()

        for i in range(len(allshapes)):
            allshapes[i].setPosition_eval([i*0.7-4, 3, allshapes[i].getPosition()[2]*2], [])
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    '''
    sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_streaming) # Initialize streaming
    while time.time()-startTime < 5:
        returnCode,data=sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_buffer) # Try to retrieve the streamed data
        if returnCode==sim.simx_return_ok: # After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
            print ('Mouse position x: ',data) # Mouse position x is actualized when the cursor is over CoppeliaSim's window
        time.sleep(0.005)
    '''


    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID,'Hello CoppeliaSim!',sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
