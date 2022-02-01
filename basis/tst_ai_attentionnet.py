import torch
import torch.nn as nn
# import net_test
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
# import net
import random

with open('test_states_target.json') as json_file:
    test_states_target = json.load(json_file)
with open('test_states.json') as json_file:
    test_states = json.load(json_file)

batch_sz = 1
n_agents = 3
n_status = 1
n_size = 1
n_color = 3
n_positions = 3
n_positions_state = 6
n_orientations = 6
n_distances = 2
action_size = n_positions + n_orientations + n_agents
n_single_state = n_color + n_size + n_positions_state + n_orientations + n_status
block_sz = n_agents * n_single_state

PATH = "state_dict_model_validation_attention_basis_001.pt"

net = attention_net.Net(batch_sz, n_agents, 0, vector_dim=128).to(device="cuda")
net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda")))
net.eval()

n_test_samples = 10
n_blocks = 3
# test_seq_len = n_blocks

test_states = torch.FloatTensor(test_states).to(device="cuda")
test_states_target = torch.FloatTensor(test_states_target).to(device="cuda")

test_states = test_states.view(n_test_samples, block_sz)
test_states_target = test_states_target.view(n_test_samples, block_sz)

if not os.path.exists('eval_results'):
    os.makedirs('eval_results')

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

    prediction_losses = []
    test_losses = []

    random_tests = 2000

    for i_batch in range(0, n_test_samples):
        shapeslist = []
        cu = 0
        cy = 0
        s = 0

        # blocks_in_game = []

        reshape = []
        arrangement = []
        timestep = []

        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

        state = test_states[i_batch, :]
        state = state.view(1, 1, block_sz)

        target = test_states_target[i_batch, :]
        target = target.view(1, 1, block_sz)

        for i in range(n_blocks):
            shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
            cu += 1

        withoutAll = []

        for idx in range(n_blocks):
            without = shapeslist.copy()
            without.remove(shapeslist[idx])
            withoutAll.append(without)

        for i_shape in range(n_blocks):
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            shape = shapeslist[i_shape]

            x = test_states[i_batch, i_shape * n_single_state + 3] * 2
            y = x
            z = x

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            sample_input = []
            sample_target = []

            r = test_states[i_batch, i_shape * n_single_state]
            g = test_states[i_batch, i_shape * n_single_state + 1]
            b1 = test_states[i_batch, i_shape * n_single_state + 2]

            fx = np.random.uniform(-1.5, 1.5)
            fy = np.random.uniform(-1.5, 1.5)

            shape.scaleShape(x, y, z)

            shape.setColor(r, g, b1)

        all_blocks = list(range(3))

        for idx in range(3):
            print(idx)
            current_loss = 1000
            current_blocks_in_game = []

            if idx != 0:
                for bl in range(0, n_blocks):
                    p_help = shapeslist[bl].getPosition()
                    if shapeslist[bl].outOfBounds(p_help):
                        status = 0
                    else:
                        status = 1
                        current_blocks_in_game.append(bl)
                    p = shapeslist[bl].get_position_basic(p_help, False, withoutAll[bl])
                    o = shapeslist[bl].getOrientationType_simple()
                    c = shapeslist[bl].getColor()
                    bb = shapeslist[bl].getBoundingBox()[0]
                    state = state.view(1, 1, block_sz)

                    state[0][0][bl * n_single_state] = c[0]
                    state[0][0][bl * n_single_state + 1] = c[1]
                    state[0][0][bl * n_single_state + 2] = c[2]
                    state[0][0][bl * n_single_state + 3] = bb
                    state[0][0][bl * n_single_state + 4] = status
                    state[0][0][bl * n_single_state + 5] = p[0]
                    state[0][0][bl * n_single_state + 6] = p[1]
                    state[0][0][bl * n_single_state + 7] = p[2]
                    state[0][0][bl * n_single_state + 8] = p[3]
                    state[0][0][bl * n_single_state + 9] = p[4]
                    state[0][0][bl * n_single_state + 10] = p[5]
                    state[0][0][bl * n_single_state + 11] = o[0]
                    state[0][0][bl * n_single_state + 12] = o[1]
                    state[0][0][bl * n_single_state + 13] = o[2]
                    state[0][0][bl * n_single_state + 14] = o[3]
                    state[0][0][bl * n_single_state + 15] = o[4]
                    state[0][0][bl * n_single_state + 16] = o[5]
                    # state[0][0][bl * state_sz + 16] = d[1]

                    state = state.to(device="cuda")

                # hidden = current_hidden

            state = state.view(1, n_blocks, n_single_state)
            # print(state)
            # print(target)

            for trial in range(random_tests):

                # actionsequence = torch.zeros([test_seq_len, 1, n_positions]).to(device="cuda")
                blocks_choice = torch.zeros([n_blocks]).to(device="cuda")

                if trial % 1000 == 0:
                    print("grid search")
                # if trial % 1 == 0:
                # print("test trial " + str(trial))
                # print("current loss: " + str(current_loss))
                current_block = torch.zeros([n_blocks])
                block_choice = np.random.choice(all_blocks)
                current_block[block_choice] = 1
                current_block = current_block.view(1, 1, n_blocks).to(device="cuda")

                po = torch.zeros([1, action_size]).to(device="cuda")

                p2 = np.random.uniform(-1, 1)
                p3 = np.random.uniform(-1, 1)
                p1 = np.random.choice([0, 1])
                possible_blocks = current_blocks_in_game.copy()
                if block_choice in possible_blocks:
                    possible_blocks.remove(block_choice)

                if not possible_blocks:
                    p0 = [-1, -1, -1]
                else:
                    p0_choice = np.random.choice(possible_blocks)
                    p0 = [0, 0, 0]
                    p0[p0_choice] = 1

                orientation_type = [1, 0, 0]
                facing_choices = [0, 0, 0]

                facing_choices[0] = np.random.uniform(- 1, 1)
                facing_choices[1] = math.sin(math.acos(facing_choices[0]))
                if random.randint(0, 1) == 0:
                    facing_choices[1] = -facing_choices[1]

                if idx == 0:
                    po[0] = torch.tensor(
                        [-1, -1, -1, 0, 0, 0, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1], facing_choices[2]])
                else:
                    po[0] = torch.tensor(
                        [p0[0], p0[1], p0[2], p1, p2, p3, orientation_type[0], orientation_type[1],
                         orientation_type[2], facing_choices[0], facing_choices[1], facing_choices[2]])

                po = po.view(1, 1, n_positions + n_orientations + n_agents).to(device="cuda")

                new_state = net(state, current_block, po, testing=True)

                loss = net.loss(new_state.view(1, block_sz), target.view(1, block_sz),
                                blocks=current_blocks_in_game, test=True, first=True)

                if loss.mean() < current_loss:
                    current_loss = loss.mean().to(device="cuda")
                    current_full_loss = loss.clone().to(device="cuda")
                    current_action = po.to(device="cuda")
                    current_best_block = current_block.to(device="cuda")
                    current_block_choice = block_choice

            # current_blocks_in_game = blocks_in_game.copy()

            if idx == 0:
                first_block = torch.argmax(current_best_block).item()
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            # current_blocks_in_game.append(torch.argmax(current_best_block).item())

            new_position = current_action.view(action_size)
            print(new_position)
            new_block = current_best_block.view(1, n_blocks).to(device="cuda")
            actionnum = torch.narrow(new_position, 0, 0, 4).view(1, 1, 4)
            orientationnum = torch.narrow(new_position, 0, n_positions + n_agents, 3).view(1, 1, 3)

            policy1 = torch.narrow(new_position, 0, 4, 2).view(1, 1, 2)
            policy2 = torch.narrow(new_position, 0, n_positions + n_agents + 3, 3).view(1, 1, 3)

            optimizer1 = torch.optim.Adam([policy1], lr=0.001)
            optimizer2 = torch.optim.Adam([policy2], lr=0.001)

            ai = AI.ActionInference(net, policy1, policy2, optimizer1, optimizer2, net.loss,
                                    attention=True)
            # print(idx)
            # print(target)

            if idx == 0:
                # current_block = blocks_target[idx].view(1, 1, n_blocks).to(device="cuda")
                action, _ = ai.action_inference(state.view(1, 1, block_sz), target.view(1, 1, block_sz),
                                                new_block, orientationnum, actionnum, current_blocks_in_game,
                                                True, current_block=current_best_block, testing=True)
            else:
                action, _ = ai.action_inference(state.view(1, 1, block_sz), target.view(1, 1, block_sz),
                                                new_block, orientationnum, actionnum, current_blocks_in_game,
                                                False, testing=True)

            action = action.view(action_size)
            prediction = net(state.view(1, n_blocks, n_single_state), new_block.view(1, n_blocks),
                             action.view(1, action_size), testing=True)

            block_nr = int(torch.argmax(new_block))
            # blocks_in_game.append(block_nr)
            # current_blocks_in_game = blocks_in_game.copy()
            ref_tensor = torch.narrow(action, 0, 0, n_agents)
            if ref_tensor[0] == -1:
                ref = -1
            else:
                ref = int(torch.argmax(ref_tensor))
            print(action)
            pos = torch.narrow(action, 0, n_agents, 3)
            ort = torch.narrow(action, 0, n_positions + n_agents, 6)

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
            op = shapeslist[block_nr].getPosition()
            shapeslist[block_nr].moveTo(2, 2, [])
            shapeslist[block_nr].setVisualOrientation_simple(ort)

            if ref == -1:
                ref_object = None
            else:
                ref_object = shapeslist[int(ref)]

            shapeslist[block_nr].set_position_basic(pos, ref_object, withoutAll[block_nr])

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(2)
            # sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            for bl in range(0, n_blocks):
                p_help = shapeslist[bl].getPosition()
                if shapeslist[bl].outOfBounds(p_help):
                    status = 0
                else:
                    status = 1
                p = shapeslist[bl].get_position_basic(p_help, False, withoutAll[bl])
                o = shapeslist[bl].getOrientationType_simple()
                c = shapeslist[bl].getColor()
                bb = shapeslist[bl].getBoundingBox()[0]
                state = state.view(1, 1, block_sz)

                state[0][0][bl * n_single_state] = c[0]
                state[0][0][bl * n_single_state + 1] = c[1]
                state[0][0][bl * n_single_state + 2] = c[2]
                state[0][0][bl * n_single_state + 3] = bb
                state[0][0][bl * n_single_state + 4] = status
                state[0][0][bl * n_single_state + 5] = p[0]
                state[0][0][bl * n_single_state + 6] = p[1]
                state[0][0][bl * n_single_state + 7] = p[2]
                state[0][0][bl * n_single_state + 8] = p[3]
                state[0][0][bl * n_single_state + 9] = p[4]
                state[0][0][bl * n_single_state + 10] = p[5]
                state[0][0][bl * n_single_state + 11] = o[0]
                state[0][0][bl * n_single_state + 12] = o[1]
                state[0][0][bl * n_single_state + 13] = o[2]
                state[0][0][bl * n_single_state + 14] = o[3]
                state[0][0][bl * n_single_state + 15] = o[4]
                state[0][0][bl * n_single_state + 16] = o[5]
                # state[0][0][bl * state_sz + 16] = d[1]

                state = state.view(1, n_blocks, n_single_state)

            print("position: " + str([ref, pos]))
            print("block: " + str(block_nr))
            print("prediction: " + str(prediction))
            print("target: " + str(target))

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        test_loss = net.loss(state.view(1, block_sz), target.view(1, block_sz))
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
            # shapeslist[i].setPosition([i-4, 3, shapeslist[i].getPosition()[2]])
            shapeslist[i].scaleShape(reshape[i][0], reshape[i][1], reshape[i][2])
            shapeslist[i].turnOriginalWayUp()

        for i in range(len(shapeslist)):
            shapeslist[i].setPosition_eval([i * 0.7 - 4, 3, shapeslist[i].getPosition()[2] * 2], [])
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    mean_prediction_loss = torch.cat([e for e in prediction_losses], -1)
    mean_test_loss = torch.cat([e for e in test_losses], -1)

    mean_prediction_loss = mean_prediction_loss.mean()
    mean_test_loss = mean_test_loss.mean()

    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID, 'Hello CoppeliaSim!', sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')
