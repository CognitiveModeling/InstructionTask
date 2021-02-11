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
        self.n_state = self.n_blocks * (self.n_type + self.n_color + self.n_boundingbox + self.n_position + self.n_orientation)
        self.batch_sz = batch_sz

        # network
        self.l1 = nn.Linear(self.n_state, self.mid_dim).to(device="cuda")
        self.L2 = nn.Linear(self.mid_dim + self.n_blocks, self.mid_dim).to(device="cuda")
        self.l3 = nn.LSTM(self.mid_dim + self.n_position + self.n_orientation, self.mid_dim, self.n_lstm_layers).to(device="cuda")
        self.l4 = nn.Linear(self.mid_dim, self.n_state).to(device="cuda")

    def init(self):
        # initialze hidden state
        hidden_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.mid_dim).to(device="cuda")
        cell_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.mid_dim).to(device="cuda")
        hidden = (hidden_state, cell_state)

        return hidden

    def forward(self, state, block, position, hidden):
        out = torch.tanh(self.l1(state))
        out = torch.cat((out, block), dim=-1)
        #out = out.view(self.n_blocks * self.batch_sz, int(self.mid_dim/self.n_blocks))
        #block = block.view(self.n_blocks * self.batch_sz, 1)
        #out = out * block
        #out = out.view(self.batch_sz, self.mid_dim)
        out = torch.tanh(self.L2(out))
        out = torch.cat((out, position), dim=-1)
        #out = out.unsqueeze(0)
        out, hidden = self.l3(out, hidden)
        out = torch.tanh(out)
        out = self.l4(out)

        return out, hidden

    def loss(self, output, target):

        #target = target.unsqueeze(0)

        loss = F.mse_loss(output, target, reduction="none")

        return loss
