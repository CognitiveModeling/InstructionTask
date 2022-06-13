import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, batch_sz, vector_dim=64):
        super(Net, self).__init__()

        self.n_blocks = 3
        self.n_position = 3
        self.n_type = 5
        self.n_orientation = 5
        self.n_action = self.n_orientation + self.n_position
        self.n_color = 3
        self.n_blocknr = 1
        self.n_boundingbox = 1
        self.n_status = 1
        self.vector_dim = vector_dim
        self.reduced_state = self.n_color + self.n_boundingbox + self.n_status
        self.single_state = self.n_color + self.n_boundingbox + self.n_orientation + self.n_status
        self.single_state_target = self.single_state + self.n_position + self.n_type
        self.single_state_target_notype = self.single_state + self.n_position
        self.n_state = self.n_blocks * self.single_state
        self.n_state_target = self.n_blocks * self.single_state_target
        self.batch_sz = batch_sz
        self.values = torch.zeros(self.batch_sz, self.n_blocks, self.vector_dim).to(device="cuda")
        self.table = torch.zeros(batch_sz, self.n_orientation + self.n_boundingbox).to(device="cuda")
        self.table_ai = torch.zeros(1, self.n_orientation + self.n_boundingbox).to(device="cuda")

        # network
        self.l1 = nn.Linear(self.n_orientation + self.n_boundingbox + self.n_action, self.vector_dim)
        self.l1b = nn.Linear(self.vector_dim, self.vector_dim)
        self.l1c = nn.Linear(self.vector_dim, self.vector_dim)
        self.l2a = nn.Linear(self.vector_dim, self.n_action)
        self.l2b = nn.Linear(self.vector_dim, 1)

        # active type
        self.type_a = nn.Sequential(nn.Linear(self.n_type, self.vector_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.vector_dim, self.vector_dim),
                                    nn.Tanh()
                                    )

        # passive type
        self.type_b = nn.Sequential(nn.Linear(self.n_type, self.vector_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.vector_dim, self.vector_dim),
                                    nn.Tanh()
                                    )

    def weight_vector(self, a, b, c, action, mask_a, mask_b, mask_c, batch_sz, types, type):
        helper = torch.zeros(batch_sz, 1).to(device="cuda")

        type_1 = self.type_a(type)

        a = torch.cat((action[:, 0], a), dim=-1)
        out_a = torch.tanh(self.l1(a))
        out_a = torch.tanh(self.l1b(out_a))
        out_a = out_a * type_1
        out_a = torch.tanh(self.l1c(out_a))
        type_a2 = self.type_b(types[:, 0])
        out_a = out_a * type_a2
        w_a = torch.sigmoid(self.l2b(out_a))
        v_a = self.l2a(out_a)

        b = torch.cat((action[:, 1], b), dim=-1)
        out_b = torch.tanh(self.l1(b))
        out_b = torch.tanh(self.l1b(out_b))
        out_b = out_b * type_1
        out_b = torch.tanh(self.l1c(out_b))
        type_b2 = self.type_b(types[:, 1])
        out_b = out_b * type_b2
        w_b = torch.sigmoid(self.l2b(out_b))
        v_b = self.l2a(out_b)

        c = torch.cat((action[:, 2], c), dim=-1)
        out_c = torch.tanh(self.l1(c))
        out_c = torch.tanh(self.l1b(out_c))
        out_c = out_c * type_1
        out_c = torch.tanh(self.l1c(out_c))
        type_c2 = self.type_b(types[:, 2])
        out_c = out_c * type_c2
        w_c = torch.sigmoid(self.l2b(out_c))
        v_c = self.l2a(out_c)

        w_a = torch.where(mask_a, w_a, helper)
        w_b = torch.where(mask_b, w_b, helper)
        w_c = torch.where(mask_c, w_c, helper)

        return w_a, w_b, w_c, v_a, v_b, v_c

    def relative_positions(self, states, action, testing=False):
        if testing:
            batch_sz = 1
        else:
            batch_sz = self.batch_sz
        relative_positions = torch.repeat_interleave(action, self.n_blocks + 1, dim=-2)
        relative_positions = relative_positions.view(batch_sz, self.n_blocks + 1, self.n_action)

        for i in range(self.n_blocks):
            relative_positions[:, i, 0] = relative_positions[:, i, 0] - states[:, i, 0]
            relative_positions[:, i, 1] = relative_positions[:, i, 1] - states[:, i, 1]

        return relative_positions

    def forward(self, states, action, type, ai=False, testing=False, printing=False):
        if ai:
            action = action.view(1, 1, self.n_action)
            states = states.view(1, self.n_blocks, self.single_state_target)

        if testing:
            batch_sz = 1
            table = self.table_ai
        else:
            batch_sz = self.batch_sz
            table = self.table

        new_action = self.relative_positions(states, action, testing)

        types = torch.narrow(states, -1, self.single_state_target_notype, self.n_type)
        states = torch.narrow(states, -1, 0, self.single_state_target_notype)

        first = torch.cat((new_action[:, self.n_blocks], table), dim=-1)
        first = torch.tanh(self.l1(first))
        first = torch.tanh(self.l1b(first))
        type_a = self.type_a(type)
        first = first * type_a
        first = torch.tanh(self.l1c(first))
        w_table = torch.sigmoid(self.l2b(first))
        v_table = self.l2a(first)

        states = states[:, :, self.n_position:(self.n_position + self.n_orientation + self.n_boundingbox)]

        a = states[:, 0]
        b = states[:, 1]
        c = states[:, 2]

        mask_a = a[:, 0] != -1
        mask_b = b[:, 0] != -1
        mask_c = c[:, 0] != -1

        w_a, w_b, w_c, v_a, v_b, v_c = self.weight_vector(a, b, c, new_action, mask_a.view(batch_sz, 1),
                                                          mask_b.view(batch_sz, 1), mask_c.view(batch_sz, 1), batch_sz,
                                                          types, type)

        residuals = w_table * v_table + w_a * v_a + w_b * v_b + w_c * v_c

        out = residuals + action

        return out

    def loss(self, output, target):

        loss = F.mse_loss(output, target, reduction="none")

        return loss
