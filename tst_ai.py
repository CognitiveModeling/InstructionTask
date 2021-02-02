import torch
import torch.nn as nn
import net_test
import json
import os
import time
import shapes
import numpy as np
import sim
import actioninference as AI
import mathown

with open('test_states_relative.json') as json_file:
        states = json.load(json_file)
with open('test_positions_relative.json') as json_file:
        positions = json.load(json_file)
with open('test_agents_relative.json') as json_file:
        agents = json.load(json_file)

n_samples = len(states)
state_sz = 21
n_blocks = 3
block_sz = state_sz * n_blocks
n_positions = 14
n_orientations = 6
seq_len = n_blocks
criterion = nn.MSELoss()
timesteps = seq_len
batch_sz = 1
n_states = seq_len + 1

random_tests = 50000

states = torch.FloatTensor(states)
positions = torch.FloatTensor(positions)
agents = torch.FloatTensor(agents)

states = torch.stack(torch.split(states, 1), dim=2) # nach Aktionsreihenfolgen geordnet
states = states.view(n_states, int(n_samples/batch_sz), batch_sz, block_sz)
agents = torch.stack(torch.split(agents, 1), dim=2)
agents = agents.view(seq_len, int(n_samples/batch_sz), batch_sz, 1)
blocks = torch.zeros([seq_len, int(n_samples/batch_sz), batch_sz, n_blocks])
positions = positions.view(int(n_samples/batch_sz), seq_len, n_positions)
positions = torch.stack(torch.split(positions, 1), dim=2)
positions = positions.view(seq_len, int(n_samples/batch_sz), batch_sz, n_positions)
for i in range(seq_len):
    for j in range(int(n_samples/batch_sz)):
        for k in range(batch_sz):
            a = int(agents[i][j][k][0])
            blocks[i][j][k][a] = 1.0

if not os.path.exists('eval_results'):
        os.makedirs('eval_results')

PATH = "state_dict_model_relative_2950samples.pt"

net = net_test.Net(batch_sz, n_blocks, timesteps)
net.load_state_dict(torch.load(PATH))
net.eval()

hidden = net.init()

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

    for i_batch in range(19, int(n_samples/batch_sz)):
        shapeslist = []
        cu = 0
        cy = 0
        s = 0

        hidden = net.init()

        for i in range(n_blocks):
            if states[seq_len, i_batch, :, i * state_sz] == 0:
                shapeslist.append(shapes.Shape(clientID, "Cuboid", cu))
                cu += 1
            if states[seq_len, i_batch, :, i * state_sz] == 1:
                shapeslist.append(shapes.Shape(clientID, "Cylinder", cy))
                cy += 1
            if states[seq_len, i_batch, :, i * state_sz] == 2:
                shapeslist.append(shapes.Shape(clientID, "Sphere", s))
                s += 1

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

        withoutAll = []

        for idx in range(n_blocks):
            without = shapeslist.copy()
            without.remove(shapeslist[idx])
            withoutAll.append(without)

        reshape = []
        arrangement = []
        timestep = []

        for i_shape in range(n_blocks):

            #shape.moveTo(shape.getPosition()[0], 0)
            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

            shape = shapeslist[i_shape]

            x = states[0, i_batch, :, i_shape * state_sz + 4] * 2
            y = states[0, i_batch, :, i_shape * state_sz + 5] * 2
            z = states[0, i_batch, :, i_shape * state_sz + 6] * 2

            xb = 1 / x
            yb = 1 / y
            zb = 1 / z

            rshape = [xb, yb, zb]
            reshape.append(rshape)

            sample_input = []
            sample_target = []

            #a = states[0, i_batch, :, i_shape * state_sz + 7]
            #b = states[0, i_batch, :, i_shape * state_sz + 8]
            #c = states[0, i_batch, :, i_shape * state_sz + 9]

            r = states[0, i_batch, :, i_shape * state_sz + 1]
            g = states[0, i_batch, :, i_shape * state_sz + 2]
            b1 = states[0, i_batch, :, i_shape * state_sz + 3]

            fx = np.random.uniform(-1.5, 1.5)
            fy = np.random.uniform(-1.5, 1.5)

            #shape.rotateX(a)
            #shape.rotateY(b)
            #shape.rotateZ(c)
            shape.scaleShape(x, y, z)

            shape.setColor(r, g, b1)

        #policy1 = torch.ones([seq_len, batch_sz, n_blocks])
        #policy2 = torch.ones([seq_len, batch_sz, n_position])

        #optimizer1 = torch.optim.Adam([policy1], lr=10)
        #optimizer2 = torch.optim.Adam([policy2], lr=0.1)

        #ai = AI.ActionInference(net, policy1, policy2, optimizer1, optimizer2, inference_cycles=10)
        #ai = AI.ActionInference(net, policy2, optimizer2, inference_cycles=10)

        time.sleep(5)

        sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)

        time.sleep(1)

        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)

        blocks_target = blocks[:, i_batch, :, :]
        positions_target = positions[:, i_batch, :, :]

        state = states[0, i_batch, :, :]
        state = state.view(1, batch_sz, block_sz)

        for idx in range(timesteps):
            print(idx)
            #sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
            target = states[idx + 1, i_batch, :, :]
            target = target.view(1, batch_sz, block_sz)

            current_block = blocks_target[idx, :, :].view(1, batch_sz, n_blocks)
            current_target_position = positions_target[idx, :, :].view(1, batch_sz, n_positions)
            first_block = np.argmax(blocks_target[0, :, :].view(n_blocks))
            first_block = first_block.item()

            current_loss = 1000
            current_position = []

            #policy1, policy2, _, hidden = ai.action_inference(state, hidden, target)
            #policy2, _, hidden = ai.action_inference(state, blocks_target, hidden, target)
            test_target, _ = net(state, current_block, current_target_position, hidden)
            test_target_loss = net.loss(test_target, target.view(1, 1, block_sz))
            #print("test loss: " + str(test_target_loss.mean()))

            for trial in range(random_tests):
                p = np.random.randint(0, 5)

                p1 = np.random.uniform(-0.3, 0.3)
                p2 = np.random.uniform(-0.3, 0.3)
                p3 = np.random.uniform(-0.3, 0.3)

                o1 = np.random.uniform(-1, 1)
                o2 = mathown.get_cos(o1)
                o3 = np.random.uniform(-1, 1)
                o4 = mathown.get_cos(o3)
                o5 = np.random.uniform(-1, 1)
                o6 = mathown.get_cos(o5)

                b = np.random.randint(0, n_blocks)

                if trial % 5000 == 0:
                    print("test trial " + str(trial))
                    print("current loss: " + str(current_loss))

                if idx == 0:
                    po = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, o1, o2, o3, o4, o5, o6])
                elif p == 0:
                    po = torch.tensor([1, 0, 0, 0, 0, p1, p2, p3, o1, o2, o3, o4, o5, o6])
                elif p == 1:
                    po = torch.tensor([0, 1, 0, 0, 0, p1, p2, p3, o1, o2, o3, o4, o5, o6])
                elif p == 2:
                    po = torch.tensor([0, 0, 1, 0, 0, p1, p2, p3, o1, o2, o3, o4, o5, o6])
                elif p == 3:
                    po = torch.tensor([0, 0, 0, 1, 0, p1, p2, p3, o1, o2, o3, o4, o5, o6])
                else:
                    po = torch.tensor([0, 0, 0, 0, 1, p1, p2, p3, o1, o2, o3, o4, o5, o6])

                po = po.view(1, batch_sz, n_positions)

                #po = current_target_position
                current, _ = net(state, current_block, po, hidden)
                loss = net.loss(current, target.view(1, 1, block_sz))

                if loss.mean() < current_loss:
                    current_loss = loss.mean()
                    current_full_loss = loss.clone()
                    current_position = po.clone()
                    current_best_block = current_block

            next_state, hidden = net(state, current_best_block, current_position, hidden)

            current_full_loss = net.loss(next_state, target.view(1, 1, block_sz))

            #if idx == 0:
            #    first_block = np.argmax(current_block.view(n_blocks)).item()
            print("loss: " + str(current_full_loss.mean()))
            print("position: " + str(current_position))
            print("block: " + str(current_best_block))
            print("first block: " + str(first_block))
            print("predicted state: " + str(next_state))
            print("target: " + str(target))
            print("full loss: " + str(current_full_loss))
            #new_position = policy2[0, 0, :]
            new_position = current_position[0, 0, :]
            #new_block = policy1[0, 0, :]
            new_block = current_best_block[0, 0, :]

            block_nr = int(torch.argmax(new_block))
            pos = torch.narrow(new_position, 0, 0, 5)
            leftright = torch.narrow(new_position, 0, 5, 1)
            frontback = torch.narrow(new_position, 0, 6, 1)
            ort = torch.narrow(new_position, 0, 8, 6)

            sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)
            op = shapeslist[block_nr].getPosition()
            shapeslist[block_nr].moveTo(2, 2, [])
            shapeslist[block_nr].setOrientation(ort)

            #time.sleep(1)

            #shapeslist[block_nr].setPosition([op[0], op[1], op[2] + 0.5], withoutAll[block_nr])
            shapeslist[block_nr].set_relative_position(pos, shapeslist[first_block], leftright, frontback, withoutAll[block_nr])

            #blocks_target[:-1] = blocks_target[1:].clone()

            sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
            time.sleep(8)
            #sim.simxPauseSimulation(clientID, sim.simx_opmode_blocking)

            for b in range(0, n_blocks):
                p_help = shapeslist[b].getPosition()
                p = shapeslist[b].getRelativePosition(shapeslist[first_block], p_help, False)
                o = shapeslist[b].getOrientation()
                c = shapeslist[b].getColor()
                bb = shapeslist[b].getBoundingBox()
                t = shapeslist[b].getType()

                state[0][0][b * state_sz] = t
                state[0][0][b * state_sz + 1] = c[0]
                state[0][0][b * state_sz + 2] = c[1]
                state[0][0][b * state_sz + 3] = c[2]
                state[0][0][b * state_sz + 4] = bb[0]
                state[0][0][b * state_sz + 5] = bb[1]
                state[0][0][b * state_sz + 6] = bb[2]
                state[0][0][b * state_sz + 7] = p[0]
                state[0][0][b * state_sz + 8] = p[1]
                state[0][0][b * state_sz + 9] = p[2]
                state[0][0][b * state_sz + 10] = p[3]
                state[0][0][b * state_sz + 11] = p[4]
                state[0][0][b * state_sz + 12] = p[5]
                state[0][0][b * state_sz + 13] = p[6]
                state[0][0][b * state_sz + 14] = p[7]
                state[0][0][b * state_sz + 15] = o[0]
                state[0][0][b * state_sz + 16] = o[1]
                state[0][0][b * state_sz + 17] = o[2]
                state[0][0][b * state_sz + 18] = o[3]
                state[0][0][b * state_sz + 19] = o[4]
                state[0][0][b * state_sz + 20] = o[5]

            print("actual state: " + str(state))

        sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
        loss = net.loss(state[0], target[0])
        print(loss)
        print(loss.mean())
        #print("state: " + str(state[0]))
        #print("target: " + str(target[0]))

        for shape in shapeslist:
            properties = []
            properties.append(shape.getType())
            properties.append(list(shape.getColor()))
            properties.append(list(shape.getBoundingBox()))
            properties.append(list(shape.getPosition()))
            properties.append(list(shape.getOrientation()))

            timestep.append(list(properties))

        arrangement.append(list(timestep))

        with open("ai3_result" + str(i_batch) + ".json", 'w') as f:
            json.dump(list(arrangement), f, indent=2)

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
