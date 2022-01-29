import torch
import torch.nn as nn
import json
import time
import shapes
import sim

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

    batch_sz = 100
    n_agents = 3
    n_status = 1
    n_size = 1
    n_color = 3
    n_positions = 4
    n_positions_state = 6
    n_orientations = 6
    n_distances = 2
    action_size = n_positions + n_orientations
    n_single_state = n_color + n_size + n_positions_state + n_orientations + n_status
    block_sz = n_agents * n_single_state

    with open('test_states_target.json') as json_file:
        target_states = json.load(json_file)
    with open('test_agents.json') as json_file:
        agents = json.load(json_file)

    for idx in range(0, 10):

        #for target in [True, False]:
        for target in [True]:
            print(idx)
            print(target)

            state_sz = 16
            n_blocks = 3
            block_sz = state_sz * n_blocks
            n_position = 9

            target_states = torch.FloatTensor(target_states)

            shapeslist = []
            cu = 0
            cy = 0
            s = 0

            for i in range(n_blocks):
                shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                cu += 1

            withoutAll = []

            for idx_b in range(n_blocks):
                without = shapeslist.copy()
                without.remove(shapeslist[idx_b])
                withoutAll.append(without)

            reshape = []
            arrangement = []
            timestep = []

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            s_list = list(range(n_blocks))

            for i_shape in s_list:
                r = target_states[idx, n_single_state * i_shape]
                g = target_states[idx, n_single_state * i_shape + 1]
                b = target_states[idx, n_single_state * i_shape + 2]

                bb1 = target_states[idx, n_single_state * i_shape + 3] * 2

                bb2 = bb1
                bb3 = bb1

                p1 = target_states[idx, n_single_state * i_shape + 5]
                p2 = target_states[idx, n_single_state * i_shape + 6]
                p3 = target_states[idx, n_single_state * i_shape + 7]
                p4 = target_states[idx, n_single_state * i_shape + 8]
                p5 = target_states[idx, n_single_state * i_shape + 9]
                p6 = target_states[idx, n_single_state * i_shape + 10]

                o1 = target_states[idx, n_single_state * i_shape + 11]
                o2 = target_states[idx, n_single_state * i_shape + 12]
                o3 = target_states[idx, n_single_state * i_shape + 13]
                o4 = target_states[idx, n_single_state * i_shape + 14]
                o5 = target_states[idx, n_single_state * i_shape + 15]
                o6 = target_states[idx, n_single_state * i_shape + 16]

                shapeslist[i_shape].setColor(r, g, b)
                shapeslist[i_shape].scaleShape(bb1, bb2, bb3)
                shapeslist[i_shape].setVisualOrientation_simple([o1, o2, o3, o4, o5, o6])
                time.sleep(1)
                if i_shape == 0:
                    shapeslist[i_shape].moveTo(0, 0, [])
                elif i_shape == 2 and p4 == 1:
                    shapeslist[i_shape].set_position_basic([p4, p5, p6], shapeslist[1], [])
                else:
                    shapeslist[i_shape].set_position_basic([p1, p2, p3], shapeslist[0], [])
            input()
            for i_shape in range(n_blocks):

                bb1 = 1/(target_states[idx, n_single_state * i_shape + 3] * 2)
                bb2 = 1/(target_states[idx, n_single_state * i_shape + 3] * 2)
                bb3 = 1/(target_states[idx, n_single_state * i_shape + 3] * 2)

                p1 = i_shape
                p2 = 3
                p3 = 0.8

                shapeslist[i_shape].scaleShape(bb1, bb2, bb3)
                shapeslist[i_shape].setPosition_eval([p1, p2, p3], withoutAll[i_shape])

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
    sim.simxAddStatusbarMessage(clientID, 'Hello CoppeliaSim!', sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')
