import net_test
import torch
import torch.nn as nn
import numpy as np
import shapes
import json

state_sz = 13
n_blocks = 3
block_sz = state_sz * n_blocks
n_positions = 6
seq_len = n_blocks
criterion = nn.MSELoss()
timesteps = seq_len
batch_sz = 1
n_states = seq_len + 1

PATH = "state_dict_model_simple_fewsamples.pt"

net = net_test.Net(batch_sz, n_blocks, timesteps)
net.load_state_dict(torch.load(PATH))
net.eval()

hidden = net.init()

with open('arrangement1354.json') as json_file:
        data = json.load(json_file)

state = torch.ones([1, 1, 1, block_sz])
target = torch.ones([1, 1, 1, block_sz])
position = torch.tensor([[[[0, 0, 0.5, 0.0000103, 0.0000032, -0.89815557]]]])
agent = torch.tensor([[[[0, 0, 1]]]])

state[0][0][0][0] = 0
state[0][0][0][13] = 0
state[0][0][0][26] = 0
state[0][0][0][1] = data[0][0][1][0]
state[0][0][0][2] = data[0][0][1][1]
state[0][0][0][3] = data[0][0][1][2]
state[0][0][0][4] = data[0][0][2][0]
state[0][0][0][5] = data[0][0][2][1]
state[0][0][0][6] = data[0][0][2][2]
state[0][0][0][7] = data[0][0][3][0]
state[0][0][0][8] = data[0][0][3][1]
state[0][0][0][9] = data[0][0][3][2]
state[0][0][0][10] = data[0][0][4][0]
state[0][0][0][11] = data[0][0][4][1]
state[0][0][0][12] = data[0][0][4][2]
state[0][0][0][14] = data[0][1][1][0]
state[0][0][0][15] = data[0][1][1][1]
state[0][0][0][16] = data[0][1][1][2]
state[0][0][0][17] = data[0][1][2][0]
state[0][0][0][18] = data[0][1][2][1]
state[0][0][0][19] = data[0][1][2][2]
state[0][0][0][20] = data[0][1][3][0]
state[0][0][0][21] = data[0][1][3][1]
state[0][0][0][22] = data[0][1][3][2]
state[0][0][0][23] = data[0][1][4][0]
state[0][0][0][24] = data[0][1][4][1]
state[0][0][0][25] = data[0][1][4][2]
state[0][0][0][27] = data[0][2][1][0]
state[0][0][0][28] = data[0][2][1][1]
state[0][0][0][29] = data[0][2][1][2]
state[0][0][0][30] = data[0][2][2][0]
state[0][0][0][31] = data[0][2][2][1]
state[0][0][0][32] = data[0][2][2][2]
state[0][0][0][33] = data[0][2][3][0]
state[0][0][0][34] = data[0][2][3][1]
state[0][0][0][35] = data[0][2][3][2]
state[0][0][0][36] = data[0][2][4][0]
state[0][0][0][37] = data[0][2][4][1]
state[0][0][0][38] = data[0][2][4][2]

target[0][0][0][0] = 0
target[0][0][0][13] = 0
target[0][0][0][26] = 0
target[0][0][0][1] = data[1][0][1][0]
target[0][0][0][2] = data[1][0][1][1]
target[0][0][0][3] = data[1][0][1][2]
target[0][0][0][4] = data[1][0][2][0]
target[0][0][0][5] = data[1][0][2][1]
target[0][0][0][6] = data[1][0][2][2]
target[0][0][0][7] = data[1][0][3][0]
target[0][0][0][8] = data[1][0][3][1]
target[0][0][0][9] = data[1][0][3][2]
target[0][0][0][10] = data[1][0][4][0]
target[0][0][0][11] = data[1][0][4][1]
target[0][0][0][12] = data[1][0][4][2]
target[0][0][0][14] = data[1][1][1][0]
target[0][0][0][15] = data[1][1][1][1]
target[0][0][0][16] = data[1][1][1][2]
target[0][0][0][17] = data[1][1][2][0]
target[0][0][0][18] = data[1][1][2][1]
target[0][0][0][19] = data[1][1][2][2]
target[0][0][0][20] = data[1][1][3][0]
target[0][0][0][21] = data[1][1][3][1]
target[0][0][0][22] = data[1][1][3][2]
target[0][0][0][23] = data[1][1][4][0]
target[0][0][0][24] = data[1][1][4][1]
target[0][0][0][25] = data[1][1][4][2]
target[0][0][0][27] = data[1][2][1][0]
target[0][0][0][28] = data[1][2][1][1]
target[0][0][0][29] = data[1][2][1][2]
target[0][0][0][30] = data[1][2][2][0]
target[0][0][0][31] = data[1][2][2][1]
target[0][0][0][32] = data[1][2][2][2]
target[0][0][0][33] = data[1][2][3][0]
target[0][0][0][34] = data[1][2][3][1]
target[0][0][0][35] = data[1][2][3][2]
target[0][0][0][36] = data[1][2][4][0]
target[0][0][0][37] = data[1][2][4][1]
target[0][0][0][38] = data[1][2][4][2]

state = state.view(1, 1, block_sz)
position = position.view(1, 1, n_positions)
agent = agent.view(1, 1, n_blocks)
target = target.view(1, 1, block_sz)

out, _ = net(state, agent, position, hidden)
loss = net.loss(out, target)
print(out)
print(target)
print(loss.mean())
