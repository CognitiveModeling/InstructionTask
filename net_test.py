import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, batch_sz, n_agents, n_actions):
        super(Net, self).__init__()

        self.n_blocks = n_agents
        self.n_position = 8
        self.n_orientation = 11
        self.n_color = 3
        self.n_blocknr = 1
        self.n_boundingbox = 3
        self.n_type = 1
        self.mid_dim = 70
        self.n_lstm_layers = 2
        self.n_hidden = 128
        self.n_state = self.n_blocks * (self.n_type + self.n_color + self.n_boundingbox + self.n_position + self.n_orientation) + 3
        self.batch_sz = batch_sz

        # network
        self.l1 = nn.Linear(self.n_state, self.n_state)
        self.l2 = nn.Linear(self.n_position + self.n_orientation, self.n_state)
        #self.l3 = nn.LSTM(self.mid_dim + self.n_position + self.n_orientation, self.mid_dim, self.n_lstm_layers)
        self.l3 = nn.Linear(self.n_blocks, self.n_state)
        self.l4 = nn.Linear(3 * self.n_state, self.n_state)
        self.l5 = nn.Linear(self.n_state, self.n_state)

    def init(self):
        # initialze hidden state
        hidden_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.mid_dim).to(device="cuda")
        cell_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.mid_dim).to(device="cuda")
        hidden = (hidden_state, cell_state)

        return hidden

    def forward(self, state, block, position):
        out1 = torch.tanh(self.l1(state))
        out2 = torch.tanh(self.l2(position))
        out3 = torch.tanh(self.l3(block))
        # out = torch.cat((out, block), dim=-1)
        # out2 = out2 * out3
        out2 = torch.cat((out2, out3), dim=-1)
        out = torch.cat((out1, out2), dim=-1)
        # out = out.unsqueeze(0)
        out = torch.tanh(self.l4(out))
        out = self.l5(out)
        out = state.detach() + out

        return out

    def loss(self, output, target):

        #target = target.squeeze(0)

        loss = F.mse_loss(output, target, reduction="none")

        return loss
