import torch
import torch.nn as nn
import net_test
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
import net

with open('test_states_relative_additional.json') as json_file:
    states = json.load(json_file)
with open('test_states_target_relative_additional.json') as json_file:
    states_target = json.load(json_file)
with open('test_positions_relative_additional.json') as json_file:
    positions = json.load(json_file)
with open('test_agents_relative_additional.json') as json_file:
    agents = json.load(json_file)


def create_base_target_state(state):
    for_cat = torch.ones(n_states, 3).to(device="cuda")
    return torch.cat((state, for_cat), dim=-1)


block_tests = 2000
action_tests = 1000
ort_tests = 1000
fine_tuning = 2000

n_samples = len(states)
n_blocks = 3
n_position = 8
seq_len = n_blocks + 2
criterion = nn.MSELoss()
timesteps = seq_len
batch_sz = 1
n_states = seq_len + 1

n_size = 1
n_color = 3
n_type = 1
n_orientations = 5
n_positions = n_position + n_orientations
n_distances = 2
action_size = n_positions + n_orientations
n_single_state = n_positions + n_distances + n_size + n_type + n_color
block_sz = n_single_state * n_blocks
state_sz = n_single_state

states = torch.FloatTensor(states).to(device="cuda")
positions = torch.FloatTensor(positions).to(device="cuda")
agents = torch.FloatTensor(agents).to(device="cuda")

states = torch.stack(torch.split(states, 1), dim=2) # nach Aktionsreihenfolgen geordnet
states = states.view(n_states, int(n_samples/batch_sz), batch_sz, block_sz)
agents = torch.stack(torch.split(agents, 1), dim=2)
agents = agents.view(seq_len, int(n_samples/batch_sz), batch_sz, 1)
blocks = torch.zeros([seq_len, int(n_samples/batch_sz), batch_sz, n_blocks]).to(device="cuda")
#positions = positions.view(int(n_samples/batch_sz), seq_len, n_positions)
positions = torch.stack(torch.split(positions, 1), dim=2)
positions = positions.view(seq_len, int(n_samples/batch_sz), batch_sz, n_positions)
for i in range(seq_len):
    for j in range(int(n_samples/batch_sz)):
        for k in range(batch_sz):
            a = int(agents[i][j][k][0])
            blocks[i][j][k][a] = 1.0

PATH = "state_dict_model_6000samples_attention_net_64_conf10_3.pt"

net = attention_net.Net(batch_sz, n_blocks, timesteps, vector_dim=32).to(device="cuda")
net.load_state_dict(torch.load(PATH, map_location=torch.device("cuda")))
net.eval()

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

    sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

    for i_batch in range(0, int(n_samples/batch_sz)):
        shapeslist = []
        cu = 0
        cy = 0
        s = 0

        #current_hidden = net.init()

        cuboid1 = shapes.Shape(clientID, "Cuboid", 0)
        cuboid2 = shapes.Shape(clientID, "Cuboid", 1)
        cuboid3 = shapes.Shape(clientID, "Cuboid", 2)
        cuboid4 = shapes.Shape(clientID, "Cuboid", 3)
        cuboid5 = shapes.Shape(clientID, "Cuboid", 4)
        sphere1 = shapes.Shape(clientID, "Sphere", 0)
        sphere2 = shapes.Shape(clientID, "Sphere", 1)
        cylinder1 = shapes.Shape(clientID, "Cylinder", 0)
        cylinder2 = shapes.Shape(clientID, "Cylinder", 1)
        cylinder3 = shapes.Shape(clientID, "Cylinder", 2)
        cylinder4 = shapes.Shape(clientID, "Cylinder", 3)
        cylinder5 = shapes.Shape(clientID, "Cylinder", 4)

        allshapes = [cuboid1, cuboid3, cuboid4, cuboid2, cuboid5, cylinder3, cylinder2, cylinder4, cylinder1, cylinder5,
                 sphere2, sphere1]

        reshape = []
        arrangement = []
        timestep = []

        #time.sleep(5)

        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

        blocks_target = blocks[:, i_batch, :, :]
        positions_target = positions[:, i_batch, :, :]

        state = states[0, i_batch, :, :]
        # state = state.view(1, batch_sz, block_sz).to(device="cuda")
        state = state.view(1, 1, block_sz)

        # target = states[timesteps, i_batch, :, :]
        # target = target.view(1, batch_sz, block_sz)
        target = states[:, i_batch, :, :]
        target = target.view(n_states, block_sz)
        ai_target = create_base_target_state(target)


        for i in range(n_blocks):
            if states[seq_len, i_batch, 0, i * n_single_state] == 0:
                shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                cu += 1
            if states[seq_len, i_batch, 0, i * n_single_state] == 1:
                shapeslist.append(shapes.Shape(clientID, "Cylinder", cy))
                cy += 1
            if states[seq_len, i_batch, 0, i * n_single_state] == 2:
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

            x = states[0, i_batch, 0, i_shape * n_single_state + 4] * 2
            y = x
            z = x

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            sample_input = []
            sample_target = []

            r = states[0, i_batch, 0, i_shape * n_single_state + 1]
            g = states[0, i_batch, 0, i_shape * n_single_state + 2]
            b1 = states[0, i_batch, 0, i_shape * n_single_state + 3]

            fx = np.random.uniform(-1.5, 1.5)
            fy = np.random.uniform(-1.5, 1.5)

            shape.scaleShape(x, y, z)

            shape.setColor(r, g, b1)

        all_blocks = list(range(3))

        for idx in range(3):

            #print(all_blocks)
            if idx != 0:
                for bl in range(0, n_blocks):
                    p_help = shapeslist[bl].getPosition()
                    p = shapeslist[bl].getRelativePosition(shapeslist[first_block], p_help, False)
                    o = shapeslist[bl].getOrientationType_simple()
                    c = shapeslist[bl].getColor()
                    bb = shapeslist[bl].getBoundingBox()[0]
                    t = shapeslist[bl].getType()
                    d = shapeslist[bl].getDistances(withoutAll[bl])
                    state = state.view(1, 1, block_sz)

                    state[0][0][bl * state_sz] = t
                    state[0][0][bl * state_sz + 1] = c[0]
                    state[0][0][bl * state_sz + 2] = c[1]
                    state[0][0][bl * state_sz + 3] = c[2]
                    state[0][0][bl * state_sz + 4] = bb
                    state[0][0][bl * state_sz + 5] = p[0]
                    state[0][0][bl * state_sz + 6] = p[1]
                    state[0][0][bl * state_sz + 7] = p[2]
                    state[0][0][bl * state_sz + 8] = p[3]
                    state[0][0][bl * state_sz + 9] = p[4]
                    state[0][0][bl * state_sz + 10] = p[5]
                    state[0][0][bl * state_sz + 11] = p[6]
                    state[0][0][bl * state_sz + 12] = p[7]
                    state[0][0][bl * state_sz + 13] = o[0]
                    state[0][0][bl * state_sz + 14] = o[1]
                    state[0][0][bl * state_sz + 15] = o[2]
                    state[0][0][bl * state_sz + 16] = o[3]
                    state[0][0][bl * state_sz + 17] = o[4]
                    state[0][0][bl * state_sz + 18] = d[0]
                    state[0][0][bl * state_sz + 19] = d[1]

                    state = state.to(device="cuda")

                #hidden = current_hidden


            #print(state)
            #print(target)
            state = state.view(1, n_blocks, n_single_state)
            #target = ai_target[seq_len].view(1, batch_sz, block_sz + 1)
            #target = ai_target[-1].view(1, batch_sz, block_sz + 1)

            current_loss = 1000

            #'''
            for trial in range(block_tests):

                # actionsequence = torch.zeros([seq_len, batch_sz, n_positions]).to(device="cuda")
                blocks_choice = torch.zeros([seq_len, batch_sz, n_blocks]).to(device="cuda")

                if trial % 1000 == 0:
                    # if trial % 1 == 0:
                    print("block trial " + str(trial))
                    print("current loss: " + str(current_loss))
                po = torch.zeros([batch_sz, n_positions]).to(device="cuda")

                p = np.random.randint(0, 5)

                p1 = np.random.uniform(-0.3, 0.3)
                p2 = np.random.uniform(-0.3, 0.3)
                p3 = np.random.uniform(-0.3, 0.3)

                current_block = torch.zeros([batch_sz, n_blocks])
                block_choice = np.random.choice(all_blocks)
                # all_blocks.remove(block_choice)
                current_block[0, block_choice] = 1
                current_block = current_block.view(1, batch_sz, n_blocks).to(device="cuda")

                orientation_type = [0, 0, 0]
                facing_choices = [0, 0]

                type_choice = np.random.randint(0, 3)

                orientation_type[type_choice] = 1

                if type_choice == 0:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                elif type_choice == 1:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                    facing = [0, 1]
                    facing_choice = np.random.choice(facing)
                    facing_choices[1] = facing_choice

                if idx == 0:
                    po[0] = torch.tensor(
                        [0, 0, 0, 0, 0, 0, 0, 0, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 0:
                    po[0] = torch.tensor(
                        [1, 0, 0, 0, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 1:
                    po[0] = torch.tensor(
                        [0, 1, 0, 0, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 2:
                    po[0] = torch.tensor(
                        [0, 0, 1, 0, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 3:
                    po[0] = torch.tensor(
                        [0, 0, 0, 1, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                else:
                    po[0] = torch.tensor(
                        [0, 0, 0, 0, 1, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])

                po = po.view(1, batch_sz, n_positions).to(device="cuda")
                # actionsequence[idx] = po
                # blocks_choice[idx] = current_block
                # po = positions_target[idx].view(1, batch_sz, n_positions).to(device="cuda")
                if idx == 0:
                    current_block = blocks_target[idx].view(1, batch_sz, n_blocks).to(device="cuda")

                new_state = net(state, current_block, po)

                # state = state.to(device="cuda")

                # del po
                # del current_block

                loss = net.loss(new_state.view(batch_sz, block_sz + 3), ai_target[seq_len].view(batch_sz, block_sz + 3))

                # print("state: " + str(state))
                # print("position: " + str(po))
                # print("block: " + str(current_block))
                # print("prediction: " + str(new_state))
                # print("target " + str(target[idx + 1].view(1, batch_sz, block_sz)))
                # print("loss: " + str(loss.mean()))

                if loss.mean() < current_loss:
                    current_loss = loss.mean().to(device="cuda")
                    current_full_loss = loss.clone().to(device="cuda")
                    # current_actionsequence = actionsequence.clone().to(device="cuda")
                    # current_best_blocks = blocks_choice.clone().to(device="cuda")
                    current_action = po.to(device="cuda")
                    current_best_block = current_block.to(device="cuda")
                    current_block_choice = block_choice
                    # current_hidden = new_hidden

            #all_blocks.remove(current_block_choice)
            #'''
            #del actionsequence
            #del block_choice
            #del loss

            final_block = current_best_block

            if idx == 0:
                first_block = torch.argmax(current_best_block).item()
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            for trial in range(action_tests):
                # actionsequence = torch.zeros([seq_len, batch_sz, n_positions]).to(device="cuda")
                blocks_choice = torch.zeros([seq_len, batch_sz, n_blocks]).to(device="cuda")

                if trial % 500 == 0:
                    # if trial % 1 == 0:
                    print("action trial " + str(trial))
                    print("current loss: " + str(current_loss))
                po = torch.zeros([batch_sz, n_positions]).to(device="cuda")

                p = np.random.randint(0, 5)

                p1 = np.random.uniform(-0.3, 0.3)
                p2 = np.random.uniform(-0.3, 0.3)
                p3 = np.random.uniform(-0.3, 0.3)

                current_block = final_block.view(batch_sz, n_blocks).to(device="cuda")

                orientation_type = [0, 0, 0]
                facing_choices = [0, 0]

                type_choice = np.random.randint(0, 3)

                orientation_type[type_choice] = 1

                if type_choice == 0:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                elif type_choice == 1:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                    facing = [0, 1]
                    facing_choice = np.random.choice(facing)
                    facing_choices[1] = facing_choice

                if idx == 0:
                    po[0] = torch.tensor(
                        [0, 0, 0, 0, 0, 0, 0, 0, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 0:
                    po[0] = torch.tensor(
                        [1, 0, 0, 0, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 1:
                    po[0] = torch.tensor(
                        [0, 1, 0, 0, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 2:
                    po[0] = torch.tensor(
                        [0, 0, 1, 0, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                elif p == 3:
                    po[0] = torch.tensor(
                        [0, 0, 0, 1, 0, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                else:
                    po[0] = torch.tensor(
                        [0, 0, 0, 0, 1, p1, p2, p3, orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])

                po = po.view(1, batch_sz, n_positions).to(device="cuda")
                # actionsequence[idx] = po
                # blocks_choice[idx] = current_block
                # po = positions_target[idx].view(1, batch_sz, n_positions).to(device="cuda")
                if idx == 0:
                    current_block = blocks_target[idx].view(1, batch_sz, n_blocks).to(device="cuda")

                new_state = net(state, current_block, po)

                loss = net.loss(new_state.view(batch_sz, block_sz + 3), ai_target[seq_len].view(batch_sz, block_sz + 3))

                if loss.mean() < current_loss:
                    current_loss = loss.mean().to(device="cuda")
                    current_full_loss = loss.clone().to(device="cuda")
                    current_action = po.to(device="cuda")

            #print(current_action)
            final_action = torch.narrow(current_action, 2, 0, 5).view(5)

            for trial in range(ort_tests):
                # actionsequence = torch.zeros([seq_len, batch_sz, n_positions]).to(device="cuda")
                blocks_choice = torch.zeros([seq_len, batch_sz, n_blocks]).to(device="cuda")

                if trial % 500 == 0:
                    # if trial % 1 == 0:
                    print("orientation trial " + str(trial))
                    print("current loss: " + str(current_loss))
                po = torch.zeros([batch_sz, n_positions]).to(device="cuda")

                p = np.random.randint(0, 5)

                p1 = np.random.uniform(-0.3, 0.3)
                p2 = np.random.uniform(-0.3, 0.3)
                p3 = np.random.uniform(-0.3, 0.3)

                current_block = final_block.view(batch_sz, n_blocks).to(device="cuda")

                orientation_type = [0, 0, 0]
                facing_choices = [0, 0]

                type_choice = np.random.randint(0, 3)

                orientation_type[type_choice] = 1

                if type_choice == 0:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                elif type_choice == 1:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                    facing = [0, 1]
                    facing_choice = np.random.choice(facing)
                    facing_choices[1] = facing_choice

                if idx == 0:
                    po[0] = torch.tensor(
                        [final_action[0], final_action[1], final_action[2], final_action[3], final_action[4], 0, 0, 0,
                         orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])
                else:
                    po[0] = torch.tensor(
                        [final_action[0], final_action[1], final_action[2], final_action[3], final_action[4], p1, p2,
                         p3,
                         orientation_type[0], orientation_type[1], orientation_type[2],
                         facing_choices[0], facing_choices[1]])

                po = po.view(1, batch_sz, n_positions).to(device="cuda")
                # actionsequence[idx] = po
                # blocks_choice[idx] = current_block
                # po = positions_target[idx].view(1, batch_sz, n_positions).to(device="cuda")
                if idx == 0:
                    current_block = blocks_target[idx].view(1, batch_sz, n_blocks).to(device="cuda")

                new_state = net(state, current_block, po)

                loss = net.loss(new_state.view(batch_sz, block_sz + 3), ai_target[seq_len].view(batch_sz, block_sz + 3))

                if loss.mean() < current_loss:
                    current_loss = loss.mean().to(device="cuda")
                    current_full_loss = loss.clone().to(device="cuda")
                    current_action = po.to(device="cuda")

            final_ort = torch.narrow(current_action, 2, 8, 3).view(3)

            for trial in range(fine_tuning):
                # actionsequence = torch.zeros([seq_len, batch_sz, n_positions]).to(device="cuda")
                blocks_choice = torch.zeros([seq_len, batch_sz, n_blocks]).to(device="cuda")

                if trial % 500 == 0:
                    # if trial % 1 == 0:
                    print("fine tuning " + str(trial))
                    print("current loss: " + str(current_loss))
                po = torch.zeros([batch_sz, n_positions]).to(device="cuda")

                p = np.random.randint(0, 5)

                p1 = np.random.uniform(-0.3, 0.3)
                p2 = np.random.uniform(-0.3, 0.3)
                p3 = np.random.uniform(-0.3, 0.3)

                current_block = final_block.view(batch_sz, n_blocks).to(device="cuda")

                orientation_type = [0, 0, 0]
                facing_choices = [0, 0]

                type_choice = np.random.randint(0, 3)

                orientation_type[type_choice] = 1

                if type_choice == 0:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                elif type_choice == 1:
                    facing_choices[0] = np.random.uniform(- 1, 1)
                    facing = [0, 1]
                    facing_choice = np.random.choice(facing)
                    facing_choices[1] = facing_choice

                if idx == 0:
                    po[0] = torch.tensor(
                        [final_action[0], final_action[1], final_action[2], final_action[3], final_action[4], 0, 0, 0,
                         final_ort[0], final_ort[1], final_ort[2], facing_choices[0], facing_choices[1]])
                else:
                    po[0] = torch.tensor(
                        [final_action[0], final_action[1], final_action[2], final_action[3], final_action[4], p1, p2,
                         p3,
                         final_ort[0], final_ort[1], final_ort[2], facing_choices[0], facing_choices[1]])

                po = po.view(1, batch_sz, n_positions).to(device="cuda")
                # actionsequence[idx] = po
                # blocks_choice[idx] = current_block
                # po = positions_target[idx].view(1, batch_sz, n_positions).to(device="cuda")
                if idx == 0:
                    current_block = blocks_target[idx].view(1, batch_sz, n_blocks).to(device="cuda")

                new_state = net(state, current_block, po)

                loss = net.loss(new_state.view(batch_sz, block_sz + 3), ai_target[seq_len].view(batch_sz, block_sz + 3))

                if loss.mean() < current_loss:
                    current_loss = loss.mean().to(device="cuda")
                    current_full_loss = loss.clone().to(device="cuda")
                    current_action = po.to(device="cuda")

            block_nr = int(torch.argmax(final_block))
            pos = torch.narrow(current_action, 2, 0, 5).view(5)
            leftright = torch.narrow(current_action, 2, 5, 1).item()
            frontback = torch.narrow(current_action, 2, 6, 1).item()
            ort = torch.narrow(current_action, 2, 8, 5).view(n_orientations)

            #print("confidence: " + str(net(state, new_block.view(1, n_blocks), new_position.view(1, n_positions))[0, 0, -1].item()))

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
            op = shapeslist[block_nr].getPosition()
            shapeslist[block_nr].moveTo(2, 2, [])
            shapeslist[block_nr].setVisualOrientation_simple(ort)

            shapeslist[block_nr].set_relative_position(pos, shapeslist[first_block], leftright, frontback, withoutAll[block_nr])

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(8)
            #sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            for bl in range(0, n_blocks):
                p_help = shapeslist[bl].getPosition()
                p = shapeslist[bl].getRelativePosition(shapeslist[first_block], p_help, False)
                o = shapeslist[bl].getOrientationType_simple()
                c = shapeslist[bl].getColor()
                bb = shapeslist[bl].getBoundingBox()[0]
                t = shapeslist[bl].getType()
                d = shapeslist[bl].getDistances(withoutAll[bl])
                state = state.view(1, 1, block_sz)

                state[0][0][bl * state_sz] = t
                state[0][0][bl * state_sz + 1] = c[0]
                state[0][0][bl * state_sz + 2] = c[1]
                state[0][0][bl * state_sz + 3] = c[2]
                state[0][0][bl * state_sz + 4] = bb
                state[0][0][bl * state_sz + 5] = p[0]
                state[0][0][bl * state_sz + 6] = p[1]
                state[0][0][bl * state_sz + 7] = p[2]
                state[0][0][bl * state_sz + 8] = p[3]
                state[0][0][bl * state_sz + 9] = p[4]
                state[0][0][bl * state_sz + 10] = p[5]
                state[0][0][bl * state_sz + 11] = p[6]
                state[0][0][bl * state_sz + 12] = p[7]
                state[0][0][bl * state_sz + 13] = o[0]
                state[0][0][bl * state_sz + 14] = o[1]
                state[0][0][bl * state_sz + 15] = o[2]
                state[0][0][bl * state_sz + 16] = o[3]
                state[0][0][bl * state_sz + 17] = o[4]
                state[0][0][bl * state_sz + 18] = d[0]
                state[0][0][bl * state_sz + 19] = d[1]

                state = state.view(1, n_blocks, n_single_state)

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            loss = net.loss(state.view(batch_sz, block_sz), target[seq_len].view(batch_sz, block_sz))
        print(loss)
        print(loss.mean())

        for shape in shapeslist:
            properties = []
            properties.append(shape.getType())
            properties.append(list(shape.getColor()))
            properties.append(list(shape.getBoundingBox()))
            properties.append(list(shape.getPosition()))
            properties.append(list(shape.getOrientation()))

            timestep.append(list(properties))

        arrangement.append(list(timestep))

        with open("ai3_result" + str(i_batch * batch_sz) + ".json", 'w') as f:
            json.dump(list(arrangement), f, indent=2)

        for i in range(len(shapeslist)):
            #shapeslist[i].setPosition([i-4, 3, shapeslist[i].getPosition()[2]])
            shapeslist[i].scaleShape(reshape[i][0], reshape[i][1], reshape[i][2])
            shapeslist[i].turnOriginalWayUp()

        for i in range(len(allshapes)):
            allshapes[i].setPosition_eval([i*0.7-4, 3, allshapes[i].getPosition()[2]*2], [])
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

        del shapeslist
        del state
        del target
        del withoutAll

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