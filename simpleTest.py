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

    cuboid = shapes.Shape(clientID, "Cuboid", 0)
    #cuboid0 = shapes.Shape(clientID, "Cuboid", 1)
    #cuboid1 = shapes.Shape(clientID, "Cuboid", 2)
    #cylinder = shapes.Shape(clientID, "Cylinder", 0)
    #cylinder0 = shapes.Shape(clientID, "Cylinder", 1)
    #sphere = shapes.Shape(clientID, "Sphere", 0)
    #sphere0 = shapes.Shape(clientID, "Sphere", 1)

    #cuboid.setAtopVariable(sphere0, 0, 0)
    #print(cuboid.getBoundingBox())

    proportion_descriptions = ["huge", "big", "large", "medium-sized", "small", "tiny",
                               "long", "broad", "bulky", "flat", "slim",
                               "horizontal", "upright", "vertical",
                                "white", "gray", "black", "brown", "yellow",
                                "red", "blue", "green", "orange", "violet",
                                "pink", "light", "dark"]

    sample = []

    for i in range(10):
        x = np.random.uniform(0.01, 0.5)
        y = np.random.uniform(0.01, 0.5)
        z = np.random.uniform(0.01, 0.5)

        xb = 1 / x
        yb = 1 / y
        zb = 1 / z

        sample_input = []
        sample_target = []

        a = np.random.uniform(0, 90)
        b = np.random.uniform(0, 90)
        c = np.random.uniform(0, 45)

        r = np.random.uniform(0, 1)
        g = np.random.uniform(0, 1)
        b1 = np.random.uniform(0, 1)

        cuboid.rotateX(a)
        cuboid.rotateY(b)
        cuboid.rotateZ(c)
        cuboid.scaleShape(x, y, z)

        cuboid.setColor(r, g, b1)
        #print(cuboid.getColor())

        sample_input.append(cuboid.shapetype_numbered)
        sample_input.append(cuboid.getBoundingBox())
        sample_input.append(cuboid.getOrientation())
        sample_input.append(cuboid.getColor())

        for idx in range(len(proportion_descriptions)):
            print("Does this property apply? ", proportion_descriptions[idx])
            answer = input()
            sample_target.append(answer)

        sample.append([sample_input, sample_target])
        print(sample[i])
        print(i+2)

        cuboid.scaleShape(xb, yb, zb)

    with open("cuboid_prop_samples_1.json", 'w') as f:
        json.dump(sample, f, indent=2)

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
