import torch
import json
import net
import random
import math

with open('states_relative.json') as json_file:
        states = json.load(json_file)
with open('positions_relative.json') as json_file:
        positions = json.load(json_file)
with open('agents_relative.json') as json_file:
        agents = json.load(json_file)

n_samples = len(states)

states = torch.FloatTensor(states)
positions = torch.FloatTensor(positions)
agents = torch.FloatTensor(agents)

batch_sz = 1
n_actions = 3
block_sz = 26 * n_actions
n_states = n_actions + 1
n_positions = 8
n_orientations = 11
action_size = n_positions + n_orientations
n_agents = 3

net = net.Net(batch_sz, n_agents, n_actions)
hidden = net.init()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

states = torch.stack(torch.split(states, 1), dim=2)# nach Aktionsreihenfolgen geordnet
states = states.view(n_states, int(n_samples/batch_sz), batch_sz, block_sz)
# sorted: (action nr, batch nr, batch, blocks)
positions = positions.view(int(n_samples/batch_sz), n_actions, action_size)
positions = torch.stack(torch.split(positions, 1), dim=2)
positions = positions.view(n_actions, int(n_samples/batch_sz), batch_sz, action_size)
agents = torch.stack(torch.split(agents, 1), dim=2)
agents = agents.view(n_actions, int(n_samples/batch_sz), batch_sz, 1)
blocks = torch.zeros([n_actions, int(n_samples/batch_sz), batch_sz, n_agents])
for i in range(n_actions):
    for j in range(int(n_samples/batch_sz)):
        for k in range(batch_sz):
            a = int(agents[i][j][k][0])
            blocks[i][j][k][a] = 1.0

batches = list(range(int(n_samples/batch_sz)))

for epoch in range(3000):
    random.shuffle(batches)
    losses = []

    for i_batch in batches:
        current_losses = []

        hidden = net.init()

        for idx in range(n_actions):
            states_batch = states[idx, i_batch, :, :]
            positions_batch = positions[idx, i_batch, :, :]
            blocks_batch = blocks[idx, i_batch, :, :]

            out, hidden = net(states_batch, blocks_batch, positions_batch, hidden)

            loss = net.loss(out, states[idx + 1, i_batch, :, :])

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

    PATH = "state_dict_model_relative_2950samples_hidden128.pt"

    # Save
    torch.save(net.state_dict(), PATH)

