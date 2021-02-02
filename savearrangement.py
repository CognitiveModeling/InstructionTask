try:
    import sim
    import shapes
    import numpy as np
    import json
    from ownfunctions import pleaseWait
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

    cuboid1 = shapes.Shape(clientID, "Cuboid", 0)
    cuboid2 = shapes.Shape(clientID, "Cuboid", 1)
    cuboid3 = shapes.Shape(clientID, "Cuboid", 2)
    # cuboid4 = shapes.Shape(clientID, "Cuboid", 3)
    # cuboid5 = shapes.Shape(clientID, "Cuboid", 4)
    # cuboid6 = shapes.Shape(clientID, "Cuboid", 5)
    sphere1 = shapes.Shape(clientID, "Sphere", 0)
    # sphere2 = shapes.Shape(clientID, "Sphere", 1)
    # sphere3 = shapes.Shape(clientID, "Sphere", 2)
    cylinder1 = shapes.Shape(clientID, "Cylinder", 0)
    cylinder2 = shapes.Shape(clientID, "Cylinder", 1)
    cylinder3 = shapes.Shape(clientID, "Cylinder", 2)

    shapes = [cuboid1, cuboid2, cuboid3, sphere1, cylinder1, cylinder2, cylinder3]

    arrangement = []

    for shape in shapes:
        properties = []
        properties.append(shape.getType())
        properties.append(shape.getColor())
        properties.append(list(shape.getBoundingBox()))
        properties.append(list(shape.getOrientation()))
        properties.append(list(shape.getPosition()))

        arrangement.append(properties)

    with open("arrangement08.json", 'w') as f:
        json.dump(list(arrangement), f, indent=2)


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