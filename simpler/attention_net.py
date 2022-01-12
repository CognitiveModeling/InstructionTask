import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, batch_sz, n_agents, n_actions, vector_dim=128):
        super(Net, self).__init__()

        self.n_blocks = n_agents
        self.n_position = 3
        self.n_position_state = 4
        self.n_orientation = 6
        self.n_distances = 2
        self.n_action = self.n_orientation + self.n_position
        self.n_color = 3
        self.n_blocknr = 1
        self.n_boundingbox = 1
        self.n_type = 1
        self.n_status = 1
        self.vector_dim = vector_dim
        self.confidence_mid_dim = 10
        #self.single_state = self.n_type + self.n_color + self.n_boundingbox + self.n_position + self.n_orientation + \
        #                    self.n_distances + self.n_status
        self.single_state = self.n_type + self.n_color + self.n_boundingbox + self.n_position_state + \
                            self.n_orientation + self.n_status
        self.n_state = self.n_blocks * self.single_state
        self.batch_sz = batch_sz
        self.values = torch.zeros(self.batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        self.for_cat = torch.zeros(self.batch_sz, 1, 3).to(device="cuda")
        self.for_cat_testing = torch.zeros(1, 1, 3).to(device="cuda")
        #self.for_cat = torch.zeros(self.batch_sz, 1, 1).to(device="cuda")

        # network
        self.q = nn.Linear(self.single_state + self.n_action, self.vector_dim)
        self.k = nn.Linear(self.single_state + self.n_action, self.vector_dim)
        self.v = nn.Linear(self.single_state + self.n_action, self.vector_dim)

        self.l1 = nn.Linear(self.vector_dim, self.vector_dim)
        self.l2 = nn.Linear(self.vector_dim, self.single_state)

        self.confidence_layer = nn.Sequential(
            nn.Linear(self.n_blocks * (self.n_action + self.single_state), self.confidence_mid_dim),
            nn.Sigmoid(),
            nn.Linear(self.confidence_mid_dim, 3),
            #nn.Linear(self.confidence_mid_dim, 1),
            nn.Sigmoid()
        )

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

    def create_base_target_state(self, state, testing=False):
        if testing:
            return torch.cat((state, self.for_cat_testing), dim=-1)
        else:
            return torch.cat((state, self.for_cat), dim=-1)

    def recreate_onehot(self, blocks):
        for i in range(self._policy2.size(0)):
            one = torch.argmax(blocks[i][0])
            u = blocks[i][0]

            for guess in range(0, len(blocks[i][0])):
                u[guess] = 0.0

            u[one] = 1.0

            blocks[i][0] = u

        return blocks

    def recreate_action(self, action, first_block):
        position = torch.narrow(action, -1, 0, 5)
        orientation = torch.narrow(action, -1, 8, 3)
        for i in range(self._policy2.size(0)):
            pos_one = torch.argmax(position[i][0])
            pos_u = position[i][0]

            ort_one = torch.argmax(orientation[i][0])
            ort_u = orientation[i][0]

            for guess in range(0, len(position[i][0])):
                pos_u[guess] = 0.0

            for guess in range(0, len(orientation[i][0])):
                ort_u[guess] = 0.0

            pos_u[pos_one] = 1.0
            ort_u[ort_one] = 1.0

            action[i][0][0] = pos_u[0]
            action[i][0][1] = pos_u[1]
            action[i][0][2] = pos_u[2]
            action[i][0][3] = pos_u[3]
            action[i][0][4] = pos_u[4]
            action[i][0][8] = ort_u[0]
            action[i][0][9] = ort_u[1]
            action[i][0][10] = ort_u[2]
            if first_block:
                for j in range(8):
                    action[i][0][j] = 0.0

        return action

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
            q[:, i, :] = self.q(block_action[:, i, :])#torch.tanh(self.q(block_action[:, i, :]))
            k[:, i, :] = self.k(block_action[:, i, :])#torch.tanh(self.k(block_action[:, i, :]))
            v[:, i, :] = self.v(block_action[:, i, :])#torch.tanh(self.v(block_action[:, i, :]))

        cos = nn.CosineSimilarity(dim=1)
        softmax = nn.Softmax(dim=2)

        d = torch.zeros(batch_sz, self.n_blocks, self.n_blocks).to(device="cuda")
        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                d[:, i, j] = cos(q[:, i, :], k[:, j, :])

        a = softmax(d)

        residuals = self.predict_values(a, v)
        #residuals = self.l1(residuals)
        residuals = torch.tanh(self.l1(residuals))
        residuals = self.l2(residuals)

        block_action = block_action.view(batch_sz, 1, self.n_blocks * (self.n_action + self.single_state))

        residuals = residuals.view(batch_sz, 1, self.n_state)
        confidence = self.confidence_layer(block_action)
        residuals = torch.cat((residuals, confidence), dim=-1)

        #print("residuals: " + str(residuals))

        states = states.view(batch_sz, 1, self.n_state)
        states = self.create_base_target_state(states, testing=testing)

        out = states + residuals

        return out

    def loss(self, output, target, blocks=None, test=False, added=False):
        if test:
            #remove duplicates
            blocks = list(dict.fromkeys(blocks))

            n = len(blocks)
            blocks.sort()
            if added:
                new_output = torch.zeros([1, n * (self.single_state - 1) + 6]).to(device="cuda")
                new_target = torch.zeros([1, n * (self.single_state - 1) + 6]).to(device="cuda")
                for b in range(n):
                    for i in range(1, self.single_state):
                        new_output[0, b * self.single_state + i - 1] = output[0, blocks[b] * self.single_state + i]
                        new_target[0, b * self.single_state + i - 1] = target[0, blocks[b] * self.single_state + i]
                    for j in range(3):
                        new_output[0, n * (self.single_state - 1) + j + 3] = output[0, self.n_state + j]
                        new_target[0, n * (self.single_state - 1) + j + 3] = target[0, self.n_state + j]
                        new_target[0, n * (self.single_state - 1) + j] = target[0, j * self.single_state]
                        new_output[0, n * (self.single_state - 1) + j] = output[0, j * self.single_state]
                    #print(blocks)
                    #print(target)
                    #print(new_target)
                    #print(output)
                    #print(new_output)

            else:
                new_output = torch.zeros([1, n * (self.single_state - 1) + 3]).to(device="cuda")
                new_target = torch.zeros([1, n * (self.single_state - 1) + 3]).to(device="cuda")
                for b in range(n):
                    for i in range(1, self.single_state):
                        new_output[0, b * self.single_state + i - 1] = output[0, blocks[b] * self.single_state + i]
                        new_target[0, b * self.single_state + i - 1] = target[0, blocks[b] * self.single_state + i]
                    for j in range(3):
                        new_target[0, n * (self.single_state - 1) + j] = target[0, j * self.single_state]
                        new_output[0, n * (self.single_state - 1) + j] = output[0, j * self.single_state]

            target = new_target
            output = new_output

        loss = F.mse_loss(output, target, reduction="none")

        return loss
