import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, batch_sz, n_agents, n_actions):
        super(Net, self).__init__()

        self.n_blocks = n_agents
        self.n_position = 8
        self.n_orientation = 11
        self.n_action = self.n_orientation + self.n_position
        self.n_color = 3
        self.n_blocknr = 1
        self.n_boundingbox = 3
        self.n_type = 1
        self.vector_dim = 128
        self.single_state = self.n_type + self.n_color + self.n_boundingbox + self.n_position + self.n_orientation
        self.n_state = self.n_blocks * self.single_state
        self.batch_sz = batch_sz
        self.values = torch.zeros(self.batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        self.for_cat = torch.zeros(self.batch_sz, 1, 1).to(device="cuda")

        # network
        self.q = nn.Linear(self.single_state + self.n_action, self.vector_dim)
        self.k = nn.Linear(self.single_state + self.n_action, self.vector_dim)
        self.v = nn.Linear(self.single_state + self.n_action, self.vector_dim)

        self.l1 = nn.Linear(self.vector_dim, self.vector_dim)
        self.l2 = nn.Linear(self.vector_dim, self.single_state)

        self.confidence_layer = nn.Linear(self.n_blocks * (self.n_action + self.single_state), 1)

        torch.autograd.set_detect_anomaly(True)

    def init(self):
        # initialze hidden state
        hidden_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.vector_dim).to(device="cuda")
        cell_state = torch.randn(self.n_lstm_layers, self.batch_sz, self.vector_dim).to(device="cuda")
        hidden = (hidden_state, cell_state)

        return hidden

    def predict_values(self, a_matrix, v_values):
        values = torch.matmul(a_matrix, v_values)
        return values

    def create_block_action(self, states, block_id, action):
        action_dummy = torch.ones(self.batch_sz, self.n_blocks, self.n_action).to(device="cuda") * -1
        block_id = torch.repeat_interleave(block_id, self.n_action, dim=-1).view(self.batch_sz, self.n_blocks, self.n_action)
        action = torch.repeat_interleave(action, 3, dim=0).view(self.batch_sz, self.n_blocks, self.n_action)
        action_dummy = torch.where(block_id == 1, action, action_dummy)
        block_action = torch.cat((states, action_dummy), dim=2)

        return block_action

    def create_base_target_state(self, state):
        return torch.cat((state, self.for_cat), dim=-1)

    def forward(self, states, block_id, action):
        block_action = self.create_block_action(states, block_id, action).to(device="cuda")

        q = torch.zeros(self.batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        k = torch.zeros(self.batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        v = torch.zeros(self.batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")

        for i in range(self.n_blocks):
            q[:, i, :] = self.q(block_action[:, i, :])#torch.tanh(self.q(block_action[:, i, :]))
            k[:, i, :] = self.k(block_action[:, i, :])#torch.tanh(self.k(block_action[:, i, :]))
            v[:, i, :] = self.v(block_action[:, i, :])#torch.tanh(self.v(block_action[:, i, :]))

        cos = nn.CosineSimilarity(dim=1)
        softmax = nn.Softmax(dim=2)

        d = torch.zeros(self.batch_sz, self.n_blocks, self.n_blocks).to(device="cuda")
        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                d[:, i, j] = cos(q[:, i, :], k[:, j, :])

        a = softmax(d)

        residuals = self.predict_values(a, v)
        #residuals = self.l1(residuals)
        residuals = torch.tanh(self.l1(residuals))
        residuals = self.l2(residuals)

        block_action = block_action.view(self.batch_sz, 1, self.n_blocks * (self.n_action + self.single_state))

        residuals = residuals.view(self.batch_sz, 1, self.n_state)
        confidence = torch.sigmoid(self.confidence_layer(block_action))
        residuals = torch.cat((residuals, confidence), dim=-1)

        #print("residuals: " + str(residuals))

        states = states.view(self.batch_sz, 1, self.n_state)
        states = self.create_base_target_state(states)

        out = states + residuals

        return out

    def loss(self, output, target):

        #target = target.unsqueeze(0)

        loss = F.mse_loss(output, target, reduction="none")

        return loss
