import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, batch_sz, n_agents, n_actions, vector_dim=32):
        super(Net, self).__init__()

        self.n_blocks = n_agents
        self.n_position = 3
        self.n_position_state = 6
        self.n_orientation = 6
        self.n_distances = 2
        self.n_action = self.n_orientation + self.n_position + n_agents
        self.n_color = 3
        self.n_blocknr = 1
        self.n_boundingbox = 1
        self.n_type = 1
        self.n_status = 1
        self.vector_dim = vector_dim
        # self.single_state = self.n_type + self.n_color + self.n_boundingbox + self.n_position + self.n_orientation + \
        #                    self.n_distances + self.n_status
        self.single_state = self.n_color + self.n_boundingbox + self.n_position_state + self.n_orientation + self.n_status
        self.n_state = self.n_blocks * self.single_state
        self.batch_sz = batch_sz
        self.values = torch.zeros(self.batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        # self.for_cat = torch.zeros(self.batch_sz, 1, 1).to(device="cuda")

        # network
        self.q = nn.Linear(self.single_state + self.n_action, self.vector_dim)
        self.k = nn.Linear(self.single_state + self.n_action, self.vector_dim)
        self.v = nn.Linear(self.single_state + self.n_action, self.vector_dim)

        self.l1 = nn.Linear(self.vector_dim, self.vector_dim)
        self.l2 = nn.Linear(self.vector_dim, self.single_state)

        #torch.autograd.set_detect_anomaly(True)

    def predict_values(self, a_matrix, v_values):
        values = torch.matmul(a_matrix, v_values)
        return values

    def create_block_action(self, states, block_id, action, testing=False):
        if testing:
            batch_sz = 1
        else:
            batch_sz = self.batch_sz
        action_dummy = torch.ones(batch_sz, self.n_blocks, self.n_action).to(device="cuda") * -1
        block_id = torch.repeat_interleave(block_id, self.n_action, dim=-1).view(batch_sz, self.n_blocks, self.n_action)
        action = torch.repeat_interleave(action, 3, dim=0).view(batch_sz, self.n_blocks, self.n_action)
        action_dummy = torch.where(block_id == 1, action, action_dummy)
        block_action = torch.cat((states, action_dummy), dim=2)

        return block_action

    def forward(self, states, block_id, action, first_block=False, ai=False, testing=False):
        if ai:
            block_id = block_id.view(1, self.n_blocks)
            action = action.view(1, self.n_action)
            states = states.view(1, self.n_blocks, self.single_state)

        if testing:
            batch_sz = 1
        else:
            batch_sz = self.batch_sz

        block_action = self.create_block_action(states, block_id, action, testing=testing).to(device="cuda")

        q = torch.zeros(batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        k = torch.zeros(batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        v = torch.zeros(batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")

        for i in range(self.n_blocks):
            q[:, i, :] = self.q(block_action[:, i, :])  # torch.tanh(self.q(block_action[:, i, :]))
            k[:, i, :] = self.k(block_action[:, i, :])  # torch.tanh(self.k(block_action[:, i, :]))
            v[:, i, :] = self.v(block_action[:, i, :])  # torch.tanh(self.v(block_action[:, i, :]))

        cos = nn.CosineSimilarity(dim=1)
        softmax = nn.Softmax(dim=2)

        d = torch.zeros(batch_sz, self.n_blocks, self.n_blocks).to(device="cuda")
        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                d[:, i, j] = cos(q[:, i, :], k[:, j, :])

        a = softmax(d)

        residuals = self.predict_values(a, v)
        # residuals = self.l1(residuals)
        residuals = torch.tanh(self.l1(residuals))
        residuals = self.l2(residuals)

        block_action = block_action.view(batch_sz, 1, self.n_blocks * (self.n_action + self.single_state))

        residuals = residuals.view(batch_sz, 1, self.n_state)

        # print("residuals: " + str(residuals))

        states = states.view(batch_sz, 1, self.n_state)

        out = states + residuals

        return out

    def loss(self, output, target, blocks=None, test=False, first=False):
        test = False
        if test:
            #remove duplicates
            blocks = list(dict.fromkeys(blocks))

            delete = [6, 7, 9, 10]

            new_output = torch.zeros([1, self.n_blocks * (self.single_state)]).to(device="cuda")
            new_target = torch.zeros([1, self.n_blocks * (self.single_state)]).to(device="cuda")

            for bl in range(self.n_blocks):
                for i in range(self.single_state):
                    if bl in blocks or i not in delete:
                        new_output[0, bl * self.single_state + i] = output[0, bl * self.single_state + i]
                        new_target[0, bl * self.single_state + i] = target[0, bl * self.single_state + i]

            #print(target)
            #print(new_target)
            #print(output)
            #print(new_output)

            target = new_target
            output = new_output

        loss = F.mse_loss(output, target, reduction="none")

        return loss
