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
max_cu = 0
max_cy = 1
max_s = 0

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


    shapeslist = []
    rands = [0, 0, 0, 1, 1, 2]
    #actionChoices = np.random.randint(8, size=n_actions)
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

    order = list(range(n_blocks))
    np.random.shuffle(order)
    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    for shape in shapeslist:

        #shape.moveTo(shape.getPosition()[0], 0)

        x = np.random.uniform(0.1, 1)
        y = np.random.uniform(0.1, 1)
        z = np.random.uniform(0.1, 1)
        x = 0.3
        #y = 0.6
        z = 1

        xb = 1 / x
        yb = 1 / y
        zb = 1 / z

        rshape = [xb, yb, zb]
        reshape.append(rshape)

        sample_input = []
        sample_target = []

        g = np.random.uniform(0, 1)
        r = np.random.uniform(0, 0.5)
        b1 = np.random.uniform(0, 0.5)
        g = 1
        r = 0
        b1 = 0

        fx = 0
        fy = 0

        #sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

        shape.scaleShape(x, y, z)

        shape.setColor(r, g, b1)

    time.sleep(1)

    sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
    shapeslist[0].moveTo(0, 0, [])

    orientation_type = [0, 0, 0, 0, 0, 0, 0, 0]
    facing_choices = [0, 0, 0]

    type = [0, 1, 2, 3, 4, 5, 6, 7]
    type_choice = np.random.choice(type)
    type_choice = 7

    orientation_type[type_choice] = 1

    if type_choice == 0 or type_choice == 1 or type_choice == 2:
        facing_choices[0] = np.random.uniform(- math.pi / 2, math.pi / 2)
    elif type_choice == 3 or type_choice == 4 or type_choice == 5:
        facing_choices[0] = np.random.uniform(- math.pi / 2, math.pi / 2)
        facing = [1, 2]
        facing_choice = np.random.choice(facing)
        facing_choices[facing_choice] = 1
    else:
        num = [1, 2]
        num_choice = np.random.choice(num)
        facing = [0, 1, 2]

        facing_choice = np.random.choice(facing, num_choice)

        if num_choice == 2:
            facing_choices[facing_choice[0]] = 1
            facing_choices[facing_choice[1]] = 1
        else:
            facing_choices[facing_choice[0]] = 1

    shapeslist[0].setVisualOrientation(
        [orientation_type[0], orientation_type[1], orientation_type[2], orientation_type[3],
         orientation_type[4], orientation_type[5], orientation_type[6], orientation_type[7],
         facing_choices[0], facing_choices[1], facing_choices[2]])
    print([orientation_type[0], orientation_type[1], orientation_type[2], orientation_type[3],
         orientation_type[4], orientation_type[5], orientation_type[6], orientation_type[7],
         facing_choices[0], facing_choices[1], facing_choices[2]])
    shapeslist[0].setPosition_eval([shapeslist[0].getPosition()[0], shapeslist[0].getPosition()[1],
                                    shapeslist[0].getBoundingBoxWorld()[2] * 0.5], [])
    #shapeslist[0].setOrientation([o1, o2, o3, o4, o5, o6])
    #rotation = shapeslist[0].rotationfromWorldAxis(alpha, beta, gamma)
    #shapeslist[0].setradianOrientation(rotation)
    #print("radian orientation: " + str(shapeslist[0].getradianOrientation()))
    #print("actual orientation: " + str(shapeslist[0].getOrientationType()))
    #shapeslist[1].setAtopCenter(shapeslist[0], withoutAll[1])
    #shapeslist[1].setColor(0, 0, 1)
    #shapeslist[2].moveRightOf(shapeslist[0], withoutAll[2])

    input()


    for i in range(len(shapeslist)):
        #shapeslist[i].setPosition([i-4, 3, shapeslist[i].getPosition()[2]])
        shapeslist[i].scaleShape(reshape[i][0], reshape[i][1], reshape[i][2])
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
