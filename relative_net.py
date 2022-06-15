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
        self.n_bounding_box = 1
        self.n_status = 1
        self.vector_dim = vector_dim
        self.single_state = self.n_color + self.n_bounding_box + self.n_orientation + self.n_status
        self.single_state_target = self.single_state + self.n_position + self.n_type
        self.single_state_target_no_type = self.single_state + self.n_position
        self.batch_sz = batch_sz
        self.table = torch.zeros(batch_sz, self.n_orientation + self.n_bounding_box).to(device="cuda")
        self.table_ai = torch.zeros(1, self.n_orientation + self.n_bounding_box).to(device="cuda")

        # base network
        self.l1 = nn.Linear(self.n_orientation + self.n_bounding_box + self.n_action, self.vector_dim)
        self.l1b = nn.Linear(self.vector_dim, self.vector_dim)
        self.l1c = nn.Linear(self.vector_dim, self.vector_dim)
        self.l2a = nn.Linear(self.vector_dim, self.n_action)
        self.l2b = nn.Linear(self.vector_dim, 1)

        # active type : type of the block being moved
        self.type_a = nn.Sequential(nn.Linear(self.n_type, self.vector_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.vector_dim, self.vector_dim),
                                    nn.Tanh()
                                    )

        # passive type : type of the block that is being looked at in relation to the active block
        self.type_b = nn.Sequential(nn.Linear(self.n_type, self.vector_dim),
                                    nn.Tanh(),
                                    nn.Linear(self.vector_dim, self.vector_dim),
                                    nn.Tanh()
                                    )

    # does a forward pass and returns values and weights for all blocks; if the block is not in the game, the weight is
    # then set to 0; same if block is being related to itself
    def weight_vector(self, a, b, c, action, mask_a, mask_b, mask_c, batch_sz, types, block_type):
        helper = torch.zeros(batch_sz, 1).to(device="cuda")

        # forward pass through active type network, encoding type of block being moved
        type_1 = self.type_a(block_type)

        # perform forward pass for action in relation to block a; receive weight and value
        a = torch.cat((action[:, 0], a), dim=-1)
        out_a = torch.tanh(self.l1(a))
        out_a = torch.tanh(self.l1b(out_a))
        out_a = out_a * type_1
        out_a = torch.tanh(self.l1c(out_a))
        type_a2 = self.type_b(types[:, 0])
        out_a = out_a * type_a2
        w_a = torch.sigmoid(self.l2b(out_a))
        v_a = self.l2a(out_a)

        # perform forward pass for action in relation to block b; receive weight and value
        b = torch.cat((action[:, 1], b), dim=-1)
        out_b = torch.tanh(self.l1(b))
        out_b = torch.tanh(self.l1b(out_b))
        out_b = out_b * type_1
        out_b = torch.tanh(self.l1c(out_b))
        type_b2 = self.type_b(types[:, 1])
        out_b = out_b * type_b2
        w_b = torch.sigmoid(self.l2b(out_b))
        v_b = self.l2a(out_b)

        # perform forward pass for action in relation to block c; receive weight and value
        c = torch.cat((action[:, 2], c), dim=-1)
        out_c = torch.tanh(self.l1(c))
        out_c = torch.tanh(self.l1b(out_c))
        out_c = out_c * type_1
        out_c = torch.tanh(self.l1c(out_c))
        type_c2 = self.type_b(types[:, 2])
        out_c = out_c * type_c2
        w_c = torch.sigmoid(self.l2b(out_c))
        v_c = self.l2a(out_c)

        # set weight to 0 if respective block is not in game or if it is the same as the block being moved
        w_a = torch.where(mask_a, w_a, helper)
        w_b = torch.where(mask_b, w_b, helper)
        w_c = torch.where(mask_c, w_c, helper)

        return w_a, w_b, w_c, v_a, v_b, v_c

    # computes target position (action) relative to the current positions of all blocks (and the tabletop)
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

    # forward pass; computes predicted resulting position of a block given a target position as well as the current
    # positions of the other blocks in the game
    def forward(self, states, action, b_type, ai=False, testing=False):
        # if called during action inference, restructure
        if ai:
            action = action.view(1, 1, self.n_action)
            states = states.view(1, self.n_blocks, self.single_state_target)

        # if called during testing, batch size is 1
        if testing:
            batch_sz = 1
            table = self.table_ai
        else:
            batch_sz = self.batch_sz
            table = self.table

        # compute target position (action) relative to the current positions of all blocks (and the tabletop)
        new_action = self.relative_positions(states, action, testing)

        # get only the type for each block
        types = torch.narrow(states, -1, self.single_state_target_no_type, self.n_type)
        # get the state without the type for each block
        states = torch.narrow(states, -1, 0, self.single_state_target_no_type)

        # do forward pass only with action in relation to tabletop; receive value (v_table) and weight (w_table)
        first = torch.cat((new_action[:, self.n_blocks], table), dim=-1)
        first = torch.tanh(self.l1(first))
        first = torch.tanh(self.l1b(first))
        type_a = self.type_a(b_type)
        first = first * type_a
        first = torch.tanh(self.l1c(first))
        w_table = torch.sigmoid(self.l2b(first))
        v_table = self.l2a(first)

        # get only rotation and size of each block
        states = states[:, :, self.n_position:(self.n_position + self.n_orientation + self.n_bounding_box)]

        # separate the blocks
        a = states[:, 0]
        b = states[:, 1]
        c = states[:, 2]
        # get a mask for each block, displaying where the tensor is != -1 (in rotation and size) -> False if block is
        # not yet in game or if block is the same as block being moved
        mask_a = a[:, 0] != -1
        mask_b = b[:, 0] != -1
        mask_c = c[:, 0] != -1

        # get values and weights for all blocks
        w_a, w_b, w_c, v_a, v_b, v_c = self.weight_vector(a, b, c, new_action, mask_a.view(batch_sz, 1),
                                                          mask_b.view(batch_sz, 1), mask_c.view(batch_sz, 1), batch_sz,
                                                          types, b_type)

        # compute the residuals; i.e. the deviation of the predicted resulting position from the target position
        residuals = w_table * v_table + w_a * v_a + w_b * v_b + w_c * v_c

        # add the residuals to the target position; receive the predicted resulting position
        out = residuals + action

        return out


# loss is simple mean squared error
def loss(output, target):

    loss = F.mse_loss(output, target, reduction="none")

    return loss
