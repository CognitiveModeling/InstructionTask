import torch
import torch.nn as nn
#import net_test
import json
import os
import time
import shapes
import numpy as np
import sim
import math
import actioninference as AI
import mathown
import attention_net
#import net
import random

with open('test_states.json') as json_file:
    test_states = json.load(json_file)
with open('test_positions.json') as json_file:
    test_positions = json.load(json_file)
with open('test_agents.json') as json_file:
    test_agents = json.load(json_file)

batch_sz = 100
n_actions = 5
n_agents = 3
n_states = n_actions + 1
n_status = 1
n_size = 1
n_color = 3
n_type = 1
n_position = 3
n_position_state = 4
n_orientations = 6
n_distances = 2
action_size = n_position + n_orientations
#n_single_state = n_position + n_orientations + n_distances + n_size + n_type + n_color + n_status
n_single_state = n_type + n_color + n_size + n_position_state + n_orientations + n_status
block_sz = n_agents * n_single_state

n_test_samples = len(test_states)
n_blocks = 3
test_seq_len = n_blocks

test_states = torch.FloatTensor(test_states).to(device="cuda")
test_positions = torch.FloatTensor(test_positions).to(device="cuda")
test_agents = torch.FloatTensor(test_agents).to(device="cuda")

test_states = torch.stack(torch.split(test_states, 1), dim=2)  # nach Aktionsreihenfolgen geordnet
test_states = test_states.view(test_seq_len + 1, int(n_test_samples / 1), 1, block_sz)
test_agents = torch.stack(torch.split(test_agents, 1), dim=2)
test_agents = test_agents.view(test_seq_len, int(n_test_samples / 1), 1, 1)
test_blocks = torch.zeros([test_seq_len, int(n_test_samples / 1), 1, n_blocks]).to(device="cuda")
# test_positions = test_positions.view(int(n_samples/1), test_seq_len, n_positions)
test_positions = torch.stack(torch.split(test_positions, 1), dim=2)
test_positions = test_positions.view(test_seq_len, int(n_test_samples / 1), 1, n_position + n_orientations)
for i in range(test_seq_len):
    for j in range(int(n_test_samples / 1)):
        for k in range(1):
            a = int(test_agents[i][j][k][0])
            test_blocks[i][j][k][a] = 1.0

if not os.path.exists('eval_results'):
        os.makedirs('eval_results')

prediction_losses = []
test_losses = []
#n_test_samples = 5
n_blocks = 3
n_position = 3
n_position_state = 4
criterion = nn.MSELoss()
timesteps = test_seq_len
batch_sz = 1
n_states = test_seq_len + 1
n_positions = n_position + n_orientations
action_size = n_positions + n_orientations
# = n_positions + n_distances + n_size + n_type + n_color + n_status
n_single_state = n_type + n_color + n_size + n_position_state + n_orientations + n_status
block_sz = n_single_state * n_blocks
state_sz = n_single_state
blocks_in_game = []

PATH = "state_dict_model_validation_attention_001.pt"

net = attention_net.Net(batch_sz, n_blocks, timesteps, vector_dim=128).to(device="cuda")
net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda")))
net.eval()


def create_base_target_state(state):
    for_cat = torch.ones(n_states, 3).to(device="cuda")
    return torch.cat((state, for_cat), dim=-1)


#hidden = net.init()
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

    i_batch = 0
    shapeslist = []
    cu = 0
    cy = 0
    s = 0

    reshape = []
    arrangement = []
    timestep = []

    #time.sleep(5)

    sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

    for i_batch in range(0, n_test_samples):
        shapeslist = []
        cu = 0
        cy = 0
        s = 0

        blocks_in_game = []

        reshape = []
        arrangement = []
        timestep = []

        #time.sleep(5)

        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

        test_blocks_target = test_blocks[:, i_batch, :, :]
        test_positions_target = test_positions[:, i_batch, :, :]

        state = test_states[0, i_batch, :, :]
        # state = state.view(1, 1, block_sz).to(device="cuda")
        state = state.view(1, 1, block_sz)

        # target = test_states[timesteps, i_batch, :, :]
        # target = target.view(1, 1, block_sz)
        target = test_states[:, i_batch, :, :]
        target = target.view(n_states, block_sz)
        ai_target = create_base_target_state(target)

        for i in range(n_blocks):
            if test_states[test_seq_len, i_batch, 0, 1 + i * state_sz] == 0:
                shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                cu += 1
            if test_states[test_seq_len, i_batch, 0, 1 + i * state_sz] == 1:
                shapeslist.append(shapes.Shape(clientID, "Cylinder", cy))
                cy += 1
            if test_states[test_seq_len, i_batch, 0, 1 + i * state_sz] == 2:
                shapeslist.append(shapes.Shape(clientID, "Sphere", s))
                s += 1

        withoutAll = []

        for idx in range(n_blocks):
            without = shapeslist.copy()
            without.remove(shapeslist[idx])
            withoutAll.append(without)

        for i_shape in range(n_blocks):
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            shape = shapeslist[i_shape]

            x = test_states[0, i_batch, 0, i_shape * state_sz + 5] * 2
            y = x
            z = x

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            sample_input = []
            sample_target = []

            r = test_states[0, i_batch, 0, i_shape * state_sz + 2]
            g = test_states[0, i_batch, 0, i_shape * state_sz + 3]
            b1 = test_states[0, i_batch, 0, i_shape * state_sz + 4]

            fx = np.random.uniform(-1.5, 1.5)
            fy = np.random.uniform(-1.5, 1.5)

            shape.scaleShape(x, y, z)

            shape.setColor(r, g, b1)

        all_blocks = list(range(3))

        for idx in range(3):
            print(idx)
            current_loss = 1000

            if idx == 1:
                all_blocks.remove(first_block)

            # print(all_blocks)
            if idx != 0:
                for bl in range(0, n_blocks):
                    p_help = shapeslist[bl].getPosition()
                    if shapeslist[bl].outOfBounds(p_help):
                        status = 0
                    else:
                        status = 1
                    p = shapeslist[bl].get_relative_position_simple(shapeslist[first_block], p_help, False,
                                                                    withoutAll[bl])
                    o = shapeslist[bl].getOrientationType_simple()
                    c = shapeslist[bl].getColor()
                    bb = shapeslist[bl].getBoundingBox()[0]
                    t = shapeslist[bl].getType()
                    d = shapeslist[bl].getDistances(withoutAll[bl])
                    state = state.view(1, 1, block_sz)

                    state[0][0][bl * state_sz] = status
                    state[0][0][bl * state_sz + 1] = t
                    state[0][0][bl * state_sz + 2] = c[0]
                    state[0][0][bl * state_sz + 3] = c[1]
                    state[0][0][bl * state_sz + 4] = c[2]
                    state[0][0][bl * state_sz + 5] = bb
                    state[0][0][bl * state_sz + 6] = p[0]
                    state[0][0][bl * state_sz + 7] = p[1]
                    state[0][0][bl * state_sz + 8] = p[2]
                    state[0][0][bl * state_sz + 9] = p[3]
                    state[0][0][bl * state_sz + 10] = o[0]
                    state[0][0][bl * state_sz + 11] = o[1]
                    state[0][0][bl * state_sz + 12] = o[2]
                    state[0][0][bl * state_sz + 13] = o[3]
                    state[0][0][bl * state_sz + 14] = o[4]
                    state[0][0][bl * state_sz + 15] = o[5]

                    state = state.to(device="cuda")

                # hidden = current_hidden

            state = state.view(1, n_blocks, n_single_state)
            # print(state)
            # print(target)

            random_tests = 1000

            for trial in range(random_tests):
                current_blocks_in_game = blocks_in_game.copy()

                # actionsequence = torch.zeros([test_seq_len, 1, n_positions]).to(device="cuda")
                blocks_choice = torch.zeros([test_seq_len, 1, n_blocks]).to(device="cuda")

                if trial % 1000 == 0:
                    print("grid search")
                #if trial % 1 == 0:
                    #print("test trial " + str(trial))
                    #print("current loss: " + str(current_loss))
                po = torch.zeros([1, n_positions]).to(device="cuda")

                #p = np.random.randint(0, 5)

                p1 = np.random.uniform(-.8, .8)
                p2 = np.random.uniform(-.8, .8)
                p3 = np.random.uniform(0, 0.5)

                current_block = torch.zeros([1, n_blocks])
                block_choice = np.random.choice(all_blocks)
                #all_blocks.remove(block_choice)
                current_block[0, block_choice] = 1
                current_block = current_block.view(1, 1, n_blocks).to(device="cuda")

                orientation_type = [0, 0, 0]
                facing_choices = [0, 0, 0]

                type_choice = np.random.randint(0, 3)

                orientation_type[type_choice] = 1

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
                    facing_choice = np.random.choice(facing)
                    facing_choices[2] = facing_choice

                if idx == 0:
                    po[0] = torch.tensor(
                        [0, 0, 0, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1], facing_choices[2]])
                else:
                    po[0] = torch.tensor(
                        [p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1], facing_choices[2]])

                po = po.view(1, 1, n_positions).to(device="cuda")
                #actionsequence[idx] = po
                #test_blocks_choice[idx] = current_block
                #po = test_positions_target[idx].view(1, 1, n_positions).to(device="cuda")
                if idx == 0:
                    current_block = test_blocks_target[idx].view(1, 1, n_blocks).to(device="cuda")

                new_state = net(state, current_block, po, testing=True)

                #state = state.to(device="cuda")

                #del po
                #del current_block

                current_blocks_in_game.append(torch.argmax(current_block).item())

                loss = net.loss(new_state.view(1, block_sz + 3), ai_target[test_seq_len].view(1, block_sz + 3),
                                blocks=current_blocks_in_game, test=True, added=True)

                #print("state: " + str(state))
                #print("position: " + str(po))
                #print("block: " + str(current_block))
                #print("prediction: " + str(new_state))
                #print("target " + str(target[idx + 1].view(1, 1, block_sz)))
                #print("loss: " + str(loss.mean()))

                if loss.mean() < current_loss:
                    current_loss = loss.mean().to(device="cuda")
                    current_full_loss = loss.clone().to(device="cuda")
                    #current_actionsequence = actionsequence.clone().to(device="cuda")
                    #current_best_blocks = blocks_choice.clone().to(device="cuda")
                    current_action = po.to(device="cuda")
                    current_best_block = current_block.to(device="cuda")
                    current_block_choice = block_choice
                    #current_hidden = new_hidden

            #all_blocks.remove(current_block_choice)

            #del actionsequence
            del block_choice
            del loss
            current_blocks_in_game = blocks_in_game.copy()

            if idx == 0:
                first_block = torch.argmax(current_best_block).item()
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            current_blocks_in_game.append(torch.argmax(current_best_block).item())

            new_position = current_action.view(n_positions)
            new_block = current_best_block.view(1, n_blocks).to(device="cuda")
            #actionnum = torch.narrow(new_position, 0, 0, n_position).view(1, 1, 5)
            orientationnum = torch.narrow(new_position, 0, n_position, 3).view(1, 1, 3)

            policy1 = torch.narrow(new_position, 0, 0, 3).view(1, 1, 3)
            policy2 = torch.narrow(new_position, 0, 6, 3).view(1, 1, 3)

            optimizer1 = torch.optim.Adam([policy1], lr=0.001)
            optimizer2 = torch.optim.Adam([policy2], lr=0.001)

            ai = AI.ActionInference(net, policy1, policy2, optimizer1, optimizer2, net.loss,
                                    attention=True)
            # print(idx)
            # print(target)

            if idx == 0:
                # current_block = blocks_target[idx].view(1, 1, n_blocks).to(device="cuda")
                action, _ = ai.action_inference(state.view(1, 1, block_sz), ai_target[-1].view(1, 1, block_sz + 3),
                                                new_block, orientationnum, current_blocks_in_game, True,
                                                current_block=current_best_block, testing=True)
            else:
                action, _ = ai.action_inference(state.view(1, 1, block_sz), ai_target[-1].view(1, 1, block_sz + 3),
                                                new_block, orientationnum, current_blocks_in_game, False,
                                                testing=True)

            action = action.view(n_positions)
            prediction = net(state.view(1, n_blocks, n_single_state), new_block.view(1, n_blocks),
                             action.view(1, n_positions), testing=True)

            block_nr = int(torch.argmax(new_block))
            blocks_in_game.append(block_nr)
            current_blocks_in_game = blocks_in_game.copy()
            pos = torch.narrow(action, 0, 0, 3)
            ort = torch.narrow(action, 0, 3, 6)

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
            op = shapeslist[block_nr].getPosition()
            shapeslist[block_nr].moveTo(2, 2, [])
            shapeslist[block_nr].setVisualOrientation_simple(ort)

            shapeslist[block_nr].set_relative_position_simple(pos, shapeslist[first_block], withoutAll[block_nr])

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(8)
            #sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            for bl in range(0, n_blocks):
                p_help = shapeslist[bl].getPosition()
                if shapeslist[bl].outOfBounds(p_help):
                    status = 0
                else:
                    status = 1
                p = shapeslist[bl].get_relative_position_simple(shapeslist[first_block], p_help, False, withoutAll[bl])
                o = shapeslist[bl].getOrientationType_simple()
                c = shapeslist[bl].getColor()
                bb = shapeslist[bl].getBoundingBox()[0]
                t = shapeslist[bl].getType()
                d = shapeslist[bl].getDistances(withoutAll[bl])
                state = state.view(1, 1, block_sz)

                state[0][0][bl * state_sz] = status
                state[0][0][bl * state_sz + 1] = t
                state[0][0][bl * state_sz + 2] = c[0]
                state[0][0][bl * state_sz + 3] = c[1]
                state[0][0][bl * state_sz + 4] = c[2]
                state[0][0][bl * state_sz + 5] = bb
                state[0][0][bl * state_sz + 6] = p[0]
                state[0][0][bl * state_sz + 7] = p[1]
                state[0][0][bl * state_sz + 8] = p[2]
                state[0][0][bl * state_sz + 9] = p[3]
                state[0][0][bl * state_sz + 10] = o[0]
                state[0][0][bl * state_sz + 11] = o[1]
                state[0][0][bl * state_sz + 12] = o[2]
                state[0][0][bl * state_sz + 13] = o[3]
                state[0][0][bl * state_sz + 14] = o[4]
                state[0][0][bl * state_sz + 15] = o[5]

                state = state.view(1, n_blocks, n_single_state)

            print("position: " + str(pos))
            print("block: " + str(block_nr))
            print("prediction: " + str(prediction))
            print("target: " + str(ai_target[-1]))

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        test_loss = net.loss(state.view(1, block_sz), target[test_seq_len].view(1, block_sz))
        test_loss = test_loss.mean().view(1)
        prediction = prediction[:, :, :block_sz]
        prediction_loss = net.loss(prediction.view(1, block_sz), state.view(1, block_sz))
        prediction_loss = prediction_loss.mean().view(1)
        if prediction_loss < 1:
            prediction_losses.append(prediction_loss)
        if test_loss < 1:
            test_losses.append(test_loss)

        input()

        for i in range(len(shapeslist)):
            #shapeslist[i].setPosition([i-4, 3, shapeslist[i].getPosition()[2]])
            shapeslist[i].scaleShape(reshape[i][0], reshape[i][1], reshape[i][2])
            shapeslist[i].turnOriginalWayUp()

        for i in range(len(shapeslist)):
            shapeslist[i].setPosition_eval([i*0.7-4, 3, shapeslist[i].getPosition()[2]*2], [])
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    mean_prediction_loss = torch.cat([e for e in prediction_losses], -1)
    mean_test_loss = torch.cat([e for e in test_losses], -1)

    mean_prediction_loss = mean_prediction_loss.mean()
    mean_test_loss = mean_test_loss.mean()
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
