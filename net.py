import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, batch_sz, n_agents, n_actions):
        super(Net, self).__init__()

        self.n_blocks = n_agents
        self.n_position = 8
        self.n_orientation = 5
        self.n_distances = 2
        self.n_color = 3
        self.n_blocknr = 1
        self.n_boundingbox = 1
        self.n_type = 1
        self.n_state = self.n_blocks * (self.n_type + self.n_color + self.n_boundingbox + self.n_position
                                        + self.n_orientation + self.n_distances)
        self.batch_sz = batch_sz

        # network
        self.l1 = nn.Linear(self.n_state, self.n_state)
        self.l2 = nn.Linear(self.n_position + self.n_orientation, self.n_state)
        #self.l3 = nn.LSTM(self.mid_dim + self.n_position + self.n_orientation, self.mid_dim, self.n_lstm_layers)
        self.l3 = nn.Linear(self.n_blocks, self.n_state)
        self.l4 = nn.Linear(3 * self.n_state, self.n_state + 1)
        self.l5 = nn.Linear(self.n_state + 1, self.n_state + 1)

    def init(self):
        # initialze hidden state
        hidden_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.mid_dim).to(device="cuda")
        cell_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.mid_dim).to(device="cuda")
        hidden = (hidden_state, cell_state)

        return hidden

    def create_base_target_state(self, state):
        for_cat = torch.zeros(self.batch_sz, 1).to(device="cuda")
        return torch.cat((state, for_cat), dim=-1)

    def recreate_onehot(self, block):
        blocks = block.clone()
        one = torch.argmax(blocks[0])
        u = blocks[0]

        for guess in range(0, len(blocks[0])):
            u[guess] = 0.0

        u[one] = 1.0

        blocks[0] = u

        return blocks

    def recreate_action(self, a, first_block):
        action = a.clone()
        position = torch.narrow(action, -1, 0, 5)
        orientation = torch.narrow(action, -1, 8, 3)

        pos_one = torch.argmax(position[0])
        pos_u = position[0]

        ort_one = torch.argmax(orientation[0])
        ort_u = orientation[0]

        for guess in range(0, len(position[0])):
            pos_u[guess] = 0.0

        for guess in range(0, len(orientation[0])):
            ort_u[guess] = 0.0

        pos_u[pos_one] = 1.0
        ort_u[ort_one] = 1.0

        action[0][0] = pos_u[0]
        action[0][1] = pos_u[1]
        action[0][2] = pos_u[2]
        action[0][3] = pos_u[3]
        action[0][4] = pos_u[4]
        action[0][8] = ort_u[0]
        action[0][9] = ort_u[1]
        action[0][10] = ort_u[2]
        if first_block:
            for j in range(8):
                action[0][j] = 0.0

        return action

    def forward(self, state, blocks, positions, first_block=False, ai=False):
        state = state.view(self.batch_sz, self.n_state)

        position = positions
        block = blocks

        out1 = torch.tanh(self.l1(state))
        out2 = torch.tanh(self.l2(position))
        out3 = torch.tanh(self.l3(block))
        #out = torch.cat((out, block), dim=-1)
        #out2 = out2 * out3
        out2 = torch.cat((out2, out3), dim=-1)
        out = torch.cat((out1, out2), dim=-1)
        out = out.unsqueeze(0)
        #out = self.l4(out)
        out = torch.tanh(self.l4(out))
        out = self.l5(out)
        state = self.create_base_target_state(state)
        out = state.detach() + out

        return out

    def loss(self, output, target):

        #target = target.view(self.batch_sz, self.n_state)

        loss = F.mse_loss(output, target, reduction="none")

        return loss
