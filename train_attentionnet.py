import torch
import json
import net
import attention_net
import random
import math

with open('states_relative_additional.json') as json_file:
    orig_states = json.load(json_file)
with open('states_target_relative_additional.json') as json_file:
    orig_states_target = json.load(json_file)
with open('positions_relative_additional.json') as json_file:
    orig_positions = json.load(json_file)
with open('agents_relative_additional.json') as json_file:
    orig_agents = json.load(json_file)

n_samples = len(orig_states)

orig_states = torch.FloatTensor(orig_states).to(device="cuda")
orig_states_target = torch.FloatTensor(orig_states_target).to(device="cuda")
orig_positions = torch.FloatTensor(orig_positions).to(device="cuda")
orig_agents = torch.FloatTensor(orig_agents).to(device="cuda")

batch_sz = 200
n_actions = 5
n_agents = 3
block_sz = 26 * n_agents
n_states = n_actions + 1
n_positions = 8
n_orientations = 11
action_size = n_positions + n_orientations
n_single_state = n_positions + n_orientations + 7

net = attention_net.Net(batch_sz, n_agents, n_actions).to(device="cuda")

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

orig_states = orig_states.view(n_samples, n_states, n_agents, n_single_state)
orig_states_target = orig_states_target.view(n_samples, n_actions, block_sz + 1)
#states = states.view(batch_sz, int(n_samples/batch_sz), n_states, block_sz)
orig_positions = orig_positions.view(n_samples, n_actions, action_size)
#positions = positions.view(batch_sz, int(n_samples/batch_sz), n_actions, action_size)
#agents = torch.stack(torch.split(agents, 1), dim=2)
orig_agents = orig_agents.view(n_samples, n_actions, 1)
#agents = agents.view(batch_sz, int(n_samples/batch_sz), n_actions, 1)
#blocks = torch.zeros([batch_sz, int(n_samples/batch_sz), n_actions, n_agents]).to(device="cuda")
#for i in range(n_actions):
#    for j in range(int(n_samples/batch_sz)):
#        for k in range(batch_sz):
#            a = int(agents[k][j][i][0])
#            blocks[k][j][i][a] = 1.0

batches = list(range(int(n_samples/batch_sz)))
samples_list = list(range(n_samples))

for epoch in range(3000):
    #random.shuffle(batches)

    #random.shuffle(samples_list)
    samples = torch.LongTensor(samples_list)
    losses = []
    states = orig_states[samples, :, :, :]
    states_target = orig_states_target[samples, :, :]
    positions = orig_positions[samples, :, :]
    agents = orig_agents[samples, :, :]

    states = states.view(batch_sz, int(n_samples / batch_sz), n_states, n_agents, n_single_state)
    states_target = states_target.view(batch_sz, int(n_samples / batch_sz), n_actions, block_sz + 1)
    positions = positions.view(batch_sz, int(n_samples / batch_sz), n_actions, action_size)
    agents = agents.view(batch_sz, int(n_samples / batch_sz), n_actions, 1)
    blocks = torch.zeros([batch_sz, int(n_samples/batch_sz), n_actions, n_agents]).to(device="cuda")
    for i in range(n_actions):
        for j in range(int(n_samples/batch_sz)):
            for k in range(batch_sz):
                a = int(agents[k][j][i][0])
                blocks[k][j][i][a] = 1.0

    for i_batch in batches:
        #print(i_batch)
        current_losses = []

        for idx in range(n_actions):
            #print(idx)
            states_batch = states[:, i_batch, idx, :, :]
            positions_batch = positions[:, i_batch, idx, :]
            blocks_batch = blocks[:, i_batch, idx, :]
            targets_batch = states_target[:, i_batch, idx, :]

            #print("states: " + str(states_batch))
            #print("positions: " + str(positions_batch))
            #print("blocks: " + str(blocks_batch))
            #print("target: " + str(targets_batch))

            out = net(states_batch, blocks_batch, positions_batch)

            loss = net.loss(out.view(batch_sz, block_sz + 1), targets_batch.view(batch_sz, block_sz + 1))

            current_losses.append(loss)

        batch_loss = torch.cat([e for e in current_losses], -1)

        batch_loss = batch_loss.mean()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        losses.append(batch_loss.view(1))

    mean_loss = torch.cat([e for e in losses], -1)
    mean_loss = mean_loss.mean()

    print(str(epoch) + ": " + str(mean_loss))

    PATH = "state_dict_model_relative_additional_6000samples_attention_net_128.pt"

    # Save
    torch.save(net.state_dict(), PATH)

