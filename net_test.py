import torch
import json
import os
import time
import shapes
import numpy as np
import sim
import math
import actioninference as AI
import relative_net
import random

n_agents = 3
n_status = 1
n_size = 1
n_color = 3
n_positions = 3
n_orientations = 5
n_type = 5
action_size = n_positions + n_orientations
n_single_state = n_color + n_size + n_orientations + n_status + n_positions + n_type
n_single_state_no_type = n_single_state - n_type
block_sz = n_agents * n_single_state


# adds margin of slack to equal function
def same_as(v1, v2):
    return v2 - 0.05 < v1 < v2 + 0.05


# arranges test samples in a way the network can read
def arrange_samples(states_batch, blocks_batch, targets_batch=None):
    batch_size = 1

    # IDs of blocks not chosen
    curr_block = torch.argmax(blocks_batch, dim=-1)
    blocks = torch.tensor([0, 1, 2]).to(device="cuda")
    blocks = blocks.repeat(batch_size, 1)
    mask = torch.ones_like(blocks).scatter_(1, curr_block.unsqueeze(1), 0.)
    blocks = blocks[mask.bool()].view(batch_size, 2)

    # ID of chosen block
    block_ids = torch.argmax(blocks_batch, dim=-1)

    # tensor filled with -1
    remaining_block_states = torch.ones(batch_size, n_agents, n_single_state).to(device="cuda") * -1

    # identifying one-hot vectors for all blocks
    block_a = torch.zeros(batch_size, n_agents).to(device="cuda")
    block_a.scatter_(1, blocks[:, 0].view(batch_size, 1), torch.ones(batch_size, n_agents - 1).to(device="cuda"))

    block_b = torch.zeros(batch_size, n_agents).to(device="cuda")
    block_b.scatter_(1, blocks[:, 1].view(batch_size, 1), torch.ones(batch_size, n_agents - 1).to(device="cuda"))

    block_id = torch.zeros(batch_size, n_agents).to(device="cuda")
    block_id.scatter_(1, block_ids.view(batch_size, 1), torch.ones(batch_size, n_agents - 1).to(device="cuda"))

    # masks with single state length, filled with 1s for all entries of the respective block
    block_a = torch.repeat_interleave(block_a.long(), n_single_state, dim=-1).view(batch_size, n_agents,
                                                                                   n_single_state)
    block_b = torch.repeat_interleave(block_b.long(), n_single_state, dim=-1).view(batch_size, n_agents,
                                                                                   n_single_state)
    block_id = torch.repeat_interleave(block_id.long(), n_single_state, dim=-1).view(batch_size, n_agents,
                                                                                     n_single_state)

    # fill the tensor with only the information of the non chosen blocks, leaving -1 for all entries of the chosen block
    remaining_block_states = torch.where(block_a == 1, states_batch, remaining_block_states)
    remaining_block_states = torch.where(block_b == 1, states_batch, remaining_block_states)

    # get only the target position for the chosen block as well as its shape type as a one-hot vector
    chosen_block_target = targets_batch[block_id == 1].view(batch_size, n_single_state)
    block_type = chosen_block_target[:, n_single_state_no_type:]
    chosen_block_target = chosen_block_target[:, :(n_positions + n_orientations)]

    return remaining_block_states, chosen_block_target, block_type


# load model and samples
PATH = "models/state_dict_model_current_mixed.pt"

net = relative_net.Net(1, vector_dim=64).to(device="cuda")
net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda")))
net.eval()

with open('datasets/test_states_target.json') as json_file:
    test_states_target = json.load(json_file)
with open('datasets/test_states.json') as json_file:
    test_states = json.load(json_file)

n_test_samples = 10
n_blocks = 3
test_seq_len = n_blocks

test_states = torch.FloatTensor(test_states).to(device="cuda")
test_states_target = torch.FloatTensor(test_states_target).to(device="cuda")

test_states = test_states.view(n_test_samples, n_agents, n_single_state)
test_states_target = test_states_target.view(n_test_samples, n_agents, n_single_state)

if not os.path.exists('eval_results'):
    os.makedirs('eval_results')

# prepare and start simulation (taken from CoppeliaSim)
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

    random_tests = 1000

    for i_batch in range(0, n_test_samples):
        shapeslist = []
        cu = 0
        cy = 0
        s = 0
        py = 0
        co = 0

        reshape = []
        arrangement = []
        timestep = []

        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

        state = test_states[i_batch, :]
        state = state.view(1, n_agents, n_single_state)

        target = test_states_target[i_batch, :]
        target = target.view(1, n_agents, n_single_state)

        # read shape types from data and select corresponding blocks in the simulation
        for i in range(n_blocks):
            if state[0, i, 13] == 1:
                shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                cu += 1
            elif state[0, i, 14] == 1:
                shapeslist.append(shapes.Shape(clientID, "Cylinder", cy))
                cy += 1
            elif state[0, i, 15] == 1:
                shapeslist.append(shapes.Shape(clientID, "Sphere", s))
                s += 1
            elif state[0, i, 16] == 1:
                shapeslist.append(shapes.Shape(clientID, "Pyramid", py))
                py += 1
            else:
                shapeslist.append(shapes.Shape(clientID, "Cone", co))
                co += 1

        withoutAll = []

        # create lists of shapes, with item i missing in each respective list

        for idx in range(n_blocks):
            without = shapeslist.copy()
            without.remove(shapeslist[idx])
            withoutAll.append(without)

        # go through list of shapes and adapt size and color according to sample data
        for i_shape in range(n_blocks):
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            shape = shapeslist[i_shape]

            x = state[0, i_shape, 8] * 2
            y = x
            z = x

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            sample_input = []
            sample_target = []

            r = state[0, i_shape, 9]
            g = state[0, i_shape, 10]
            b1 = state[0, i_shape, 11]

            fx = np.random.uniform(-1.5, 1.5)
            fy = np.random.uniform(-1.5, 1.5)

            shape.scale_shape(x, y, z)

            shape.set_color(r, g, b1)

        all_blocks = list(range(3))

        for idx in range(3):
            print(idx)
            current_loss = 1000
            if idx == 0:
                current_blocks_not_in_game = all_blocks.copy() # if first trial, all blocks are not in game yet

            else:
                current_blocks_not_in_game = []
                # read current game state, including positions of all blocks and determine which are in the game
                for bl in range(0, n_blocks):
                    p_help = shapeslist[bl].get_raw_position()
                    if shapes.out_of_bounds(p_help):
                        status = 0
                        current_blocks_not_in_game.append(bl)
                    else:
                        status = 1
                    p = shapeslist[bl].get_position_clean()
                    o = shapeslist[bl].get_orientation_type_simple()
                    c = shapeslist[bl].get_color()
                    bb = shapeslist[bl].get_bounding_box()[0]
                    t = shapeslist[bl].get_type()
                    b_type = [0, 0, 0, 0, 0]
                    b_type[t] = 1
                    if bl == 0:
                        state = torch.zeros(1, 1, block_sz).to(device="cuda")
                    p[2] = p[2] - bb * 0.5
                    if same_as(p[2], 0):
                        p[2] = 0

                    state[0][0][bl * n_single_state] = p[0]
                    state[0][0][bl * n_single_state + 1] = p[1]
                    state[0][0][bl * n_single_state + 2] = p[2]
                    state[0][0][bl * n_single_state + 3] = o[0]
                    state[0][0][bl * n_single_state + 4] = o[1]
                    state[0][0][bl * n_single_state + 5] = o[2]
                    state[0][0][bl * n_single_state + 6] = o[3]
                    state[0][0][bl * n_single_state + 7] = o[4]
                    state[0][0][bl * n_single_state + 8] = bb
                    state[0][0][bl * n_single_state + 9] = c[0]
                    state[0][0][bl * n_single_state + 10] = c[1]
                    state[0][0][bl * n_single_state + 11] = c[2]
                    state[0][0][bl * n_single_state + 12] = status
                    state[0][0][bl * n_single_state + 13] = b_type[0]
                    state[0][0][bl * n_single_state + 14] = b_type[1]
                    state[0][0][bl * n_single_state + 15] = b_type[2]
                    state[0][0][bl * n_single_state + 16] = b_type[3]
                    state[0][0][bl * n_single_state + 17] = b_type[4]

            state = state.view(1, n_blocks, n_single_state)

            # grid search: perform predictions for randomly generated actions, always keep the one action with
            # the best prediction
            for trial in range(random_tests):
                blocks_choice = torch.zeros([n_blocks]).to(device="cuda")

                if trial % 1000 == 0:
                    print("grid search")

                # choose block to be moved, position, and orientation for next move
                current_block = torch.zeros([n_blocks])
                block_choice = np.random.choice(current_blocks_not_in_game)
                current_block[block_choice] = 1
                current_block = current_block.view(1, 1, n_blocks).to(device="cuda")

                p1 = np.random.uniform(-.05, .05)
                p2 = np.random.uniform(-.05, .05)
                p3 = np.random.uniform(0, 1)

                p1 = target[0, block_choice, 0] + p1
                p2 = target[0, block_choice, 1] + p2

                po = torch.zeros(action_size)

                orientation_type = [1, 0, 0]
                facing_choices = [0, 0, 0]

                facing_choices[0] = np.random.uniform(- 1, 1)
                facing_choices[1] = math.sin(math.acos(facing_choices[0]))
                if random.randint(0, 1) == 0:
                    facing_choices[1] = -facing_choices[1]

                po = torch.tensor([p1, p2, p3, orientation_type[0], orientation_type[1],
                                   orientation_type[2], facing_choices[0], facing_choices[1]])

                po = po.view(1, action_size).to(device="cuda")
                current_block = current_block.view(1, n_agents).to(device="cuda")

                # adapt sample data to fit network
                other_block_states, this_block_target, b_type = arrange_samples(state, current_block, target)

                # compute predicted future state via network
                new_state = net(other_block_states, po, b_type, testing=True)

                loss = relative_net.loss(new_state.view(1, action_size), this_block_target.view(1, action_size))

                # check if loss is smaller than all previous losses; if so, save current data
                if loss.mean() < current_loss:
                    current_loss = loss.mean().to(device="cuda")
                    current_full_loss = loss.clone().to(device="cuda")
                    current_action = po.to(device="cuda")
                    current_best_block = current_block.to(device="cuda")
                    current_block_choice = block_choice

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            # adapt best data once again to fit network
            other_block_states, this_block_target, b_type = arrange_samples(state, current_best_block, target)

            new_position = current_action.view(action_size)
            new_block = current_best_block.view(1, n_blocks).to(device="cuda")
            orientation_num = torch.narrow(new_position, 0, n_positions, 3).view(1, 1, 3)

            # set action inference policies to be adapted, choosing action determined by grid search
            policy1 = torch.narrow(new_position, 0, 0, n_positions).view(1, 1, n_positions)
            policy2 = torch.narrow(new_position, 0, n_positions + 3, 2).view(1, 1, 2)

            # set action inference optimizers
            optimizer1 = torch.optim.Adam([policy1], lr=0.001)
            optimizer2 = torch.optim.Adam([policy2], lr=0.001)

            # define action inference object
            ai = AI.ActionInference(net, policy1, policy2, optimizer1, optimizer2, relative_net.loss)

            # execute action inference
            action, _ = ai.action_inference(other_block_states.view(1, 1, block_sz),
                                            this_block_target.view(1, 1, action_size),
                                            orientation_num, b_type, testing=True)

            action = action.view(1, action_size)

            # compute prediction of definite action create by action inference
            prediction = net(other_block_states.view(1, 1, n_agents, n_single_state),
                             action.view(1, 1, action_size), b_type, ai=True, testing=True, printing=True)

            block_nr = int(torch.argmax(new_block))
            action = action.view(action_size)
            pos = torch.narrow(action, 0, 0, n_positions)
            pos_old = pos.clone()
            ort = torch.narrow(action, 0, n_positions, n_orientations)

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
            op = shapeslist[block_nr].get_position_clean()
            shapeslist[block_nr].move_to(2, 2, [])
            shapeslist[block_nr].set_visual_orientation_simple(ort)

            if pos[2] < 0:
                pos[2] = 0

            pos[2] = pos[2] + shapeslist[block_nr].get_bounding_box()[0] * 0.5

            # set position as determined by action inference
            shapeslist[block_nr].set_position(pos, withoutAll[block_nr])

            print("prev state: " + str(state))
            print("position: " + str(pos_old))
            print("new pos: " + str(action))
            print("prediction: " + str(prediction))
            print("block: " + str(block_nr))
            print("target: " + str(target))

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(2)

            # get resulting state of simulation
            for bl in range(0, n_blocks):
                p_help = shapeslist[bl].get_raw_position()
                if shapes.out_of_bounds(p_help):
                    status = 0
                    current_blocks_not_in_game.append(bl)
                else:
                    status = 1
                p = shapeslist[bl].get_position_clean()
                o = shapeslist[bl].get_orientation_type_simple()
                c = shapeslist[bl].get_color()
                bb = shapeslist[bl].get_bounding_box()[0]
                t = shapeslist[bl].get_type()
                b_type = [0, 0, 0, 0, 0]
                b_type[t] = 1
                if bl == 0:
                    state = torch.zeros(1, 1, block_sz).to(device="cuda")
                p[2] = p[2] - bb * 0.5
                if same_as(p[2], 0):
                    p[2] = 0

                state[0][0][bl * n_single_state] = p[0]
                state[0][0][bl * n_single_state + 1] = p[1]
                state[0][0][bl * n_single_state + 2] = p[2]
                state[0][0][bl * n_single_state + 3] = o[0]
                state[0][0][bl * n_single_state + 4] = o[1]
                state[0][0][bl * n_single_state + 5] = o[2]
                state[0][0][bl * n_single_state + 6] = o[3]
                state[0][0][bl * n_single_state + 7] = o[4]
                state[0][0][bl * n_single_state + 8] = bb
                state[0][0][bl * n_single_state + 9] = c[0]
                state[0][0][bl * n_single_state + 10] = c[1]
                state[0][0][bl * n_single_state + 11] = c[2]
                state[0][0][bl * n_single_state + 12] = status
                state[0][0][bl * n_single_state + 13] = b_type[0]
                state[0][0][bl * n_single_state + 14] = b_type[1]
                state[0][0][bl * n_single_state + 15] = b_type[2]
                state[0][0][bl * n_single_state + 16] = b_type[3]
                state[0][0][bl * n_single_state + 17] = b_type[4]

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

        # reset all blocks
        for i in range(len(shapeslist)):
            shapeslist[i].scale_shape(reshape[i][0], reshape[i][1], reshape[i][2])
            shapeslist[i].turn_original_way_up()
            shapeslist[i].set_position_eval([i * 4 - 4, 3.3, 1])

        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID, 'Hello CoppeliaSim!', sim.simx_opmode_oneshot)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive.
    # You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')
