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

n_blocks = 3
n_actions = n_blocks
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

    even_list = list(range(0, 5000))

    for j in even_list:
        print(j)

        shapeslist = []
        rands = np.random.randint(3, size=n_blocks)
        actionChoices = np.random.randint(8, size=n_actions)
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

        for shape in shapeslist:

            #shape.moveTo(shape.getPosition()[0], 0)
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            x = np.random.uniform(0.1, 1)
            y = np.random.uniform(0.1, 1)
            z = np.random.uniform(0.1, 1)

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            sample_input = []
            sample_target = []

            a = np.random.uniform(0, 90)
            b = np.random.uniform(0, 90)
            c = np.random.uniform(0, 45)

            r = np.random.uniform(0, 1)
            g = np.random.uniform(0, 1)
            b1 = np.random.uniform(0, 1)

            #fx = np.random.uniform(-1.5, 1.5)
            #fy = np.random.uniform(-1.5, 1.5)
            fx = 0
            fy = 0

            #shape.rotateX(a)
            #shape.rotateY(b)
            #shape.rotateZ(c)
            shape.scaleShape(x, y, z)

            shape.setColor(r, g, b1)
            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

        time.sleep(1)

        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

        time.sleep(1)
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

        for a in range(n_actions):
            timestep = []

            for shape in shapeslist:

                if a == 0:
                    properties = []
                    properties.append(shape.getType())
                    properties.append(shape.getColor())
                    properties.append(list(shape.getBoundingBox()))
                    properties.append(list(shape.getRelativePosition(shapeslist[order[0]], shape.getPosition(), first=True)))
                    properties.append(list(shape.getOrientationType()))

                    timestep.append(list(properties))
                else:
                    properties = []
                    properties.append(shape.getType())
                    properties.append(shape.getColor())
                    properties.append(list(shape.getBoundingBox()))
                    properties.append(list(shape.getRelativePosition(shapeslist[order[0]], shape.getPosition(), first=False)))
                    properties.append(list(shape.getOrientationType()))

                    timestep.append(list(properties))

            orientation_type = [0, 0, 0, 0, 0, 0, 0, 0]
            facing_choices = [0, 0, 0]

            type = [0, 1, 2, 3, 4, 5, 6, 7]
            type_choice = np.random.choice(type)

            orientation_type[type_choice] = 1

            if type_choice == 0 or type_choice == 1 or type_choice == 2:
                facing_choices[0] = np.random.uniform(- math.pi/2, math.pi/2)
            elif type_choice == 3 or type_choice == 4 or type_choice == 5:
                facing_choices[0] = np.random.uniform(- math.pi/2, math.pi/2)
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
            '''
            if j % 4 == 0:
                o1 = float(np.random.choice(choice))
                o5 = 0
                o2 = mathown.get_cos(o1)
                o3 = np.random.uniform(-1, 1)
                o4 = mathown.get_cos(o3)
                o6 = mathown.get_cos(o5)
            else:
                o1 = 0
                o5 = np.random.uniform(-1, 1)
                o2 = mathown.get_cos(o1)
                o3 = 0
                o4 = mathown.get_cos(o3)
                o6 = mathown.get_cos(o5)
            '''

            if a == 0:
                shapeslist[order[a]].setVisualOrientation([orientation_type[0], orientation_type[1], orientation_type[2], orientation_type[3],
                                        orientation_type[4], orientation_type[5], orientation_type[6], orientation_type[7],
                                        facing_choices[0], facing_choices[1], facing_choices[2]])
                time.sleep(1)
                shapeslist[order[a]].moveTo(fx, fy, [])
                #orientation = list([o1, o2, o3, o4, o5, o6])
                #print(list([o1, o2, o3, o4, o5, o6]))
                orientation = shapeslist[order[a]].getOrientationType()
                #print(orientation)
                position = [0, 0, 0, 0, 0, 0, 0, 0]

            else:
                leftright = np.random.uniform(-0.25, 0.25)
                frontback = np.random.uniform(-0.25, 0.25)
                shapeslist[order[a]].moveTo(2, 2, [])
                shapeslist[order[a]].setVisualOrientation([orientation_type[0], orientation_type[1], orientation_type[2], orientation_type[3],
                                        orientation_type[4], orientation_type[5], orientation_type[6], orientation_type[7],
                                        facing_choices[0], facing_choices[1], facing_choices[2]])
                #sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
                #orientation = list([o1, o2, o3, o4, o5, o6])
                orientation = shapeslist[order[a]].getOrientationType()
                position = shapeslist[order[a]].performRandomActionVariable(shapeslist[order[0]], actionChoices[a], leftright, frontback, withoutAll[order[a]])
                #position = shapeslist[order[a]].performRandomActionVariable(shapeslist[order[0]], 3, leftright, frontback, withoutAll[order[a]])

            time.sleep(2)
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(3)

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            properties = []
            properties.append([order[a]])
            properties.append(list(position))
            properties.append(orientation)

            timestep.append(list(properties))

            arrangement.append(list(timestep))

        timestep = []

        for shape in shapeslist:
            properties = []
            properties.append(shape.getType())
            properties.append(list(shape.getColor()))
            properties.append(list(shape.getBoundingBox()))
            properties.append(list(shape.getRelativePosition(shapeslist[order[0]], shape.getPosition(), first=False)))
            properties.append(list(shape.getOrientationType()))

            timestep.append(list(properties))

        arrangement.append(list(timestep))

        with open("arrangement" + str(j) + ".json", 'w') as f:
            json.dump(list(arrangement), f, indent=2)
        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

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
