# Make sure to have the server side running in CoppeliaSim: 
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!
import random
import csv

import mathown

try:
    import sim
    import shapes
    import numpy as np
    import json
    import math
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time

print ('Program started')
sim.simxFinish(-1) # just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim

n_blocks = 1
n_actions = n_blocks
max_cu = 3
max_cy = 2
max_s = 2
max_co = 2
max_py = 2

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

colors_prot = [black, white, red, green, yellow, blue, brown, purple, pink, orange, gray]
colors_nonprot = [blueish, greenish, reddish, yellowish, brownish]

if clientID!=-1:
    print ('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res,objs=sim.simxGetObjects(clientID,sim.sim_handle_all,sim.simx_opmode_blocking)
    if res==sim.simx_return_ok:
        print('Number of objects in the scene: ', len(objs))
    else:
        print('Remote API function call returned with error code: ',res)

    time.sleep(2)

    # Now retrieve streaming data (i.e. in a non-blocking fashion):
    startTime=time.time()

    shapeslist = []
    rands = [4, 0, 1, 2, 3]
    #actionChoices = np.random.randint(8, size=n_actions)
    cu = 0
    cy = 0
    s = 0
    co = 0
    py = 0

    for i in range(n_blocks):
        if rands[i] == 0:
            if cu < max_cu:
                shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                cu += 1
            else:
                rands[i] = 1
        elif rands[i] == 1:
            if cy < max_cy:
                shapeslist.append(shapes.Shape(clientID, "Cylinder", cy))
                cy += 1
            else:
                rands[i] = 2
        elif rands[i] == 2:
            if s < max_s:
                shapeslist.append(shapes.Shape(clientID, "Sphere", s))
                s += 1
        elif rands[i] == 3:
            if co < max_co:
                shapeslist.append(shapes.Shape(clientID, "Cone", s))
                co += 1
        else:
            if py < max_py:
                shapeslist.append(shapes.Shape(clientID, "Pyramid", s))
                py += 1

    sample = []
    withoutAll = []

    for n in range(n_blocks):
        withoutn = shapeslist.copy()
        withoutn.remove(shapeslist[n])
        withoutAll.append(withoutn)

    reshape = []
    arrangement = []

    order = list(range(n_blocks))
    np.random.shuffle(order)
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    #shapeslist[0].setOrientation([o1, o2, o3, o4, o5, o6])
    #rotation = shapeslist[0].rotationfromWorldAxis(alpha, beta, gamma)
    #shapeslist[0].setradianOrientation(rotation)
    #print("radian orientation: " + str(shapeslist[0].getradianOrientation()))
    #print("actual orientation: " + str(shapeslist[0].getOrientationType()))
    #shapeslist[1].setAtopCenter(shapeslist[0], withoutAll[1])
    #shapeslist[1].setColor(0, 0, 1)
    #shapeslist[2].moveRightOf(shapeslist[0], withoutAll[2])

    for i in range(401, 501):
        print(i)

        for shape in shapeslist:
            # shape.moveTo(shape.getPosition()[0], 0)

            x = random.uniform(0.2, 1.)
            y = x
            z = x

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            color_type = random.choice([0, 0, 1, 1, 1])

            if color_type == 1:
                rgb = random.choice(colors_prot)
            else:
                rgb = random.choice(colors_nonprot)

            r, g, b1 = rgb

            sample_input = []
            sample_target = []

            # sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            shape.scaleShape(x, y, z)

            shape.setColor(r, g, b1)

            time.sleep(1)

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            values = {
                "size": shape.getBoundingBox()[0],
                "r": r,
                "g": g,
                "b": b1,
                "type": shape.getType(),
                "id": i,
                "color_type": color_type
            }

            a = 0.
            b = 0.
            '''
            orientation_type = [0, 0, 0]
            facing_choices = [0, 0]

            type = [0, 0, 0, 0, 0, 1, 1, 1, 2]
            type_choice = np.random.choice(type)

            orientation_type[type_choice] = 1

            if type_choice == 0:
                facing_choices[0] = np.random.uniform(- 1, 1)
            elif type_choice == 1:
                facing_choices[0] = np.random.uniform(- 1, 1)
                facing_choices[1] = np.random.choice([0, 1])

            shape.setVisualOrientation_simple(
                [orientation_type[0], orientation_type[1], orientation_type[2], facing_choices[0], facing_choices[1]])
            #print([orientation_type[0], orientation_type[1], orientation_type[2], facing_choices[0], facing_choices[1]])
            '''

            #time.sleep(1)

            shape.moveTo(a, b, [])

            #print(shape.getOrientationType_simple())

            #'''
            with open('attribute_files.csv', mode='a') as csv_file:
                fieldnames = ['id', 'type', 'size', 'r', 'g', 'b', 'color_type']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                if i == 1:
                    writer.writeheader()
                writer.writerow(values)
            #'''
        input()

        for i in range(len(shapeslist)):
            #shapeslist[i].setPosition([i-4, 3, shapeslist[i].getPosition()[2]])
            shapeslist[i].scaleShape(reshape[i][0], reshape[i][1], reshape[i][2])
            reshape = []
    #time.sleep(1)
    #sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    #sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
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
