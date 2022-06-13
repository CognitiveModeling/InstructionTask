import torch
import json
import attention_net
import random

# trains the net 'attention net' with the data from "datasets"

batch_sz = 200
n_agents = 3
n_status = 1
n_size = 1
n_color = 3
n_positions = 3
n_orientations = 5
n_type = 5
action_size = n_positions + n_orientations
n_single_state = n_color + n_size + n_orientations + n_status + n_positions + n_type
n_single_state_no_type = n_color + n_size + n_orientations + n_status + n_positions
block_sz = n_agents * n_single_state


# arranges the samples in a form the model can read
def arrange_samples(states_batch, blocks_batch, targets_batch):
    current_block = torch.argmax(blocks_batch, dim=-1)
    blocks = torch.tensor([0, 1, 2]).to(device="cuda")
    blocks = blocks.repeat(batch_sz, 1)
    mask = torch.ones_like(blocks).scatter_(1, current_block.unsqueeze(1), 0.)
    blocks = blocks[mask.bool()].view(batch_sz, 2)

    block_ids = torch.argmax(blocks_batch, dim=-1)

    other_block_states = torch.ones(batch_sz, n_agents, n_single_state).to(device="cuda") * -1

    block_a = torch.zeros(batch_sz, n_agents).to(device="cuda")
    block_a.scatter_(1, blocks[:, 0].view(batch_sz, 1), torch.ones(batch_sz, n_agents - 1).to(device="cuda"))

    block_b = torch.zeros(batch_sz, n_agents).to(device="cuda")
    block_b.scatter_(1, blocks[:, 1].view(batch_sz, 1), torch.ones(batch_sz, n_agents - 1).to(device="cuda"))

    block_id = torch.zeros(batch_sz, n_agents).to(device="cuda")
    block_id.scatter_(1, block_ids.view(batch_sz, 1), torch.ones(batch_sz, n_agents - 1).to(device="cuda"))

    block_a = torch.repeat_interleave(block_a.long(), n_single_state, dim=-1).view(batch_sz, n_agents,
                                                                                   n_single_state)
    block_b = torch.repeat_interleave(block_b.long(), n_single_state, dim=-1).view(batch_sz, n_agents,
                                                                                   n_single_state)
    block_id = torch.repeat_interleave(block_id.long(), n_single_state, dim=-1).view(batch_sz, n_agents,
                                                                                     n_single_state)

    other_block_states = torch.where(block_a == 1, states_batch, other_block_states)
    other_block_states = torch.where(block_b == 1, states_batch, other_block_states)

    this_block_target = targets_batch[block_id == 1].view(batch_sz, n_single_state)
    b_type = this_block_target[:, n_single_state_no_type:]
    this_block_target = this_block_target[:, :(n_positions + n_orientations)]

    return other_block_states, this_block_target, b_type


with open('datasets/states.json') as json_file:
    orig_states = json.load(json_file)
with open('datasets/states_target.json') as json_file:
    orig_states_target = json.load(json_file)
with open('datasets/positions.json') as json_file:
    orig_positions = json.load(json_file)
with open('datasets/agents.json') as json_file:
    orig_agents = json.load(json_file)

validation_sz = 1600
n_samples = (7600 + 8400) * 5
n_train_samples = n_samples - validation_sz

orig_states = torch.FloatTensor(orig_states).to(device="cuda")
orig_states_target = torch.FloatTensor(orig_states_target).to(device="cuda")
orig_positions = torch.FloatTensor(orig_positions).to(device="cuda")
orig_agents = torch.FloatTensor(orig_agents).to(device="cuda")

# network
net = attention_net.Net(batch_sz, vector_dim=64).to(device="cuda")

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

orig_states = orig_states.view(n_samples, n_agents, n_single_state)
orig_states_target = orig_states_target.view(n_samples, n_agents, n_single_state)
orig_positions = orig_positions.view(n_samples, action_size)
orig_agents = orig_agents.view(n_samples, 1)

batches = list(range(int(n_train_samples / batch_sz)))
samples_list = list(range(n_train_samples))
validation_samples_list = list(range(n_train_samples, n_samples))

for epoch in range(1000):

    # get samples
    random.shuffle(samples_list)
    samples = torch.LongTensor(samples_list)
    validation_samples = torch.LongTensor(validation_samples_list)
    losses = []
    validation_losses = []
    states = orig_states[samples, :, :]
    states_target = orig_states_target[samples, :]
    positions = orig_positions[samples, :]
    agents = orig_agents[samples, :]

    validation_states = orig_states[validation_samples, :, :]
    validation_states_target = orig_states_target[validation_samples, :]
    validation_positions = orig_positions[validation_samples, :]
    validation_agents = orig_agents[validation_samples, :]

    states = states.view(batch_sz, int(n_train_samples / batch_sz), n_agents, n_single_state)
    states_target = states_target.view(batch_sz, int(n_train_samples / batch_sz), n_agents, n_single_state)
    positions = positions.view(batch_sz, int(n_train_samples / batch_sz), action_size)
    agents = agents.view(batch_sz, int(n_train_samples / batch_sz), 1)
    blocks = torch.zeros([batch_sz, int(n_train_samples / batch_sz), n_agents]).to(device="cuda")
    for j in range(int(n_train_samples / batch_sz)):
        for k in range(batch_sz):
            a = int(agents[k][j][0])
            blocks[k][j][a] = 1.0

    validation_states = validation_states.view(batch_sz, int(validation_sz / batch_sz), n_agents, n_single_state)
    validation_states_target = validation_states_target.view(batch_sz, int(validation_sz / batch_sz),
                                                             n_agents, n_single_state)
    validation_positions = validation_positions.view(batch_sz, int(validation_sz / batch_sz), action_size)
    validation_agents = validation_agents.view(batch_sz, int(validation_sz / batch_sz), 1)
    validation_blocks = torch.zeros([batch_sz, int(validation_sz / batch_sz), n_agents]).to(device="cuda")
    for j in range(int(validation_sz / batch_sz)):
        for k in range(batch_sz):
            a = int(validation_agents[k][j][0])
            validation_blocks[k][j][a] = 1.0

    # start training

    for i_batch in batches:
        states_batch = states[:, i_batch, :, :]
        positions_batch = positions[:, i_batch, :]
        blocks_batch = blocks[:, i_batch, :]
        targets_batch = states_target[:, i_batch, :]

        # reformat samples
        other_block_states, this_block_target, b_type = arrange_samples(states_batch, blocks_batch, targets_batch)

        out = net(other_block_states, positions_batch, b_type)

        loss = net.loss(out.view(batch_sz, (n_positions + n_orientations)),
                        this_block_target.view(batch_sz, (n_positions + n_orientations)))

        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.view(1))

    mean_loss = torch.cat([e for e in losses], -1)
    mean_loss = mean_loss.mean()

    # validation

    validation_states_batch = validation_states[:, 0, :, :]
    validation_positions_batch = validation_positions[:, 0, :]
    validation_blocks_batch = validation_blocks[:, 0, :]
    validation_targets_batch = validation_states_target[:, 0, :]

    other_block_states, this_block_target, b_type = arrange_samples(validation_states_batch, validation_blocks_batch,
                                                                    validation_targets_batch)

    out = net(other_block_states, validation_positions_batch, b_type)

    validation_loss = net.loss(out.view(batch_sz, (n_positions + n_orientations)),
                               this_block_target.view(batch_sz, (n_positions + n_orientations)))

    validation_loss = validation_loss.mean()

    print(str(epoch) + " training loss: " + str(mean_loss.item()) + " validation loss: " + str(
        validation_loss.item()))

    PATH = "models/state_dict_model_current_mixed.pt"

    # Save
    torch.save(net.state_dict(), PATH)
