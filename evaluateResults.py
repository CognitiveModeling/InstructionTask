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

    state_sz = 16
    n_blocks = 3
    block_sz = state_sz * n_blocks
    n_position = 9
    criterion = nn.MSELoss()
    timesteps = 20
    batch_sz = 1
    n_states = n_blocks + 1
    seq_len = n_blocks + 2

    for idx in range(0, 10):
        with open('test_states_relative_additional.json') as json_file:
            target_states = json.load(json_file)
        with open('test_agents_relative_additional.json') as json_file:
            target_blocks = json.load(json_file)
        with open('ai3_result_attention' + str(idx) + '.json') as json_file:
            ai_states = json.load(json_file)

        helper = []

        for j in range(n_blocks):
            helper.append(ai_states[0][j][0])
            for m in range(4):
                if m + 1 == 4:
                    for n in range(6):
                        helper.append(ai_states[0][j][m + 1][n])
                else:
                    for n in range(3):
                        helper.append(ai_states[0][j][m + 1][n])

        #for target in [True, False]:
        for target in [True]:
            print(idx)
            print(target)

            state_sz = 16
            n_blocks = 3
            block_sz = state_sz * n_blocks
            n_position = 9

            target_states = torch.FloatTensor(target_states)
            ai_states = torch.FloatTensor(helper)

            ai_states = ai_states.view([1, 1, block_sz])

            first_block = target_blocks[idx][0][0]

            if target:
                states = target_states[idx, seq_len, :]

                n_samples = len(states)
                n_blocks = 3
                n_positions = 8
                seq_len = n_blocks + 2
                criterion = nn.MSELoss()
                timesteps = seq_len
                batch_sz = 1
                n_states = seq_len + 1

                n_size = 1
                n_color = 3
                n_type = 1
                n_orientations = 5
                n_position = n_positions + n_orientations
                n_distances = 2
                action_size = n_positions + n_orientations
                n_single_state = n_position + n_distances + n_size + n_type + n_color
                block_sz = n_single_state * n_blocks
                state_sz = n_single_state
            else:
                states = ai_states[0, 0, :]

            shapeslist = []
            cu = 0
            cy = 0
            s = 0

            for i in range(n_blocks):
                #print(states[i * state_sz])
                if states[i * state_sz] == 0:
                    shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                    cu += 1
                if states[i * state_sz] == 1:
                    shapeslist.append(shapes.Shape(clientID, "Cylinder", cy))
                    cy += 1
                if states[i * state_sz] == 2:
                    shapeslist.append(shapes.Shape(clientID, "Sphere", s))
                    s += 1

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
            s_list.remove(first_block)
            s_list.insert(0, first_block)

            for i_shape in s_list:
                r = states[state_sz * i_shape + 1]
                g = states[state_sz * i_shape + 2]
                b = states[state_sz * i_shape + 3]

                bb1 = states[state_sz * i_shape + 4] * 2
                if not target:
                    bb2 = states[state_sz * i_shape + 5] * 2
                    bb3 = states[state_sz * i_shape + 6] * 2

                    p1 = states[state_sz * i_shape + 7]
                    p2 = states[state_sz * i_shape + 8]
                    p3 = states[state_sz * i_shape + 9]

                    o1 = states[state_sz * i_shape + 10]
                    o2 = states[state_sz * i_shape + 11]
                    o3 = states[state_sz * i_shape + 12]
                    o4 = states[state_sz * i_shape + 13]
                    o5 = states[state_sz * i_shape + 14]
                    o6 = states[state_sz * i_shape + 15]
                if target:
                    bb2 = bb1
                    bb3 = bb1

                    p1 = states[state_sz * i_shape + 5]
                    p2 = states[state_sz * i_shape + 6]
                    p3 = states[state_sz * i_shape + 7]
                    p4 = states[state_sz * i_shape + 8]
                    p5 = states[state_sz * i_shape + 9]
                    p6 = states[state_sz * i_shape + 10]
                    p7 = states[state_sz * i_shape + 11]
                    p8 = states[state_sz * i_shape + 12]

                    o1 = states[state_sz * i_shape + 13]
                    o2 = states[state_sz * i_shape + 14]
                    o3 = states[state_sz * i_shape + 15]
                    o4 = states[state_sz * i_shape + 16]
                    o5 = states[state_sz * i_shape + 17]

                shapeslist[i_shape].setColor(r, g, b)
                shapeslist[i_shape].scaleShape(bb1, bb2, bb3)
                if not target:
                    shapeslist[i_shape].setOrientation([o1, o2, o3, o4, o5, o6])
                    shapeslist[i_shape].setPosition_eval([p1, p2, p3], [])
                else:
                    shapeslist[i_shape].setVisualOrientation_simple([o1, o2, o3, o4, o5])
                    time.sleep(1)
                    shapeslist[i_shape].setPosition_eval_from_relativePosition([p1, p2, p3, p4, p5], shapeslist[first_block], p6, p7, p8, [])
            input()
            for i_shape in range(n_blocks):

                bb1 = 1/(states[state_sz * i_shape + 4] * 2)
                bb2 = 1/(states[state_sz * i_shape + 4] * 2)
                bb3 = 1/(states[state_sz * i_shape + 4] * 2)

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
