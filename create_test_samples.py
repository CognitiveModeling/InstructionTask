import json
import math


# adds margin of slack to equal function
def same_as(v1, v2):
    return v2 - 0.05 < v1 < v2 + 0.05


r = list(range(0, 10))
# random.shuffle(r)

n_agents = 3
n_actions = 3
states = []
states_target = []
agents = []
positions = []

for i in r:
    old_position = [[], [], []]
    position_t = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    action = [0, 0, 0]
    reference_block = -1
    orig_ref_block = -1
    old_next_position = [[], [], []]
    with open('test_arrangements/arrangement' + str(i) + '.json') as json_file:
        data = json.load(json_file)

        for j in range(n_actions):
            position = []
            next_position = []
            orientation = []
            next_orientation = []
            color = []
            size = []
            in_game = []
            in_game_next = []
            block_type = []

            # get all information for all blocks
            for l in range(n_agents):
                state = []
                state_target = []
                all_blocks = list(range(n_agents))
                all_blocks.remove(l)
                color.append(data[j][l][2])
                size.append(data[j][l][3])
                in_game.append(data[j][l][4])
                in_game_next.append(data[j + 1][l][4])
                position.append(data[j][l][0])
                if same_as(position[l][2], 0):
                    position[l][2] = 0
                orientation.append(data[j][l][1])
                next_position.append(data[j + 1][l][0])
                if same_as(next_position[l][2], 0):
                    next_position[l][2] = 0
                next_orientation.append(data[j + 1][l][1])
                type_helper = [0, 0, 0, 0, 0]
                type_helper[data[j][l][5]] = 1
                block_type.append(type_helper)

            block_to_be_moved = data[j][n_agents][0][0]
            target_position = data[j][n_agents][1]
            target_orientation = data[j][n_agents][2]

            state = [[position[0][0], position[0][1], position[0][2], orientation[0][0], orientation[0][1],
                      orientation[0][2], orientation[0][3], orientation[0][4], size[0][0], color[0][0], color[0][1],
                      color[0][2], in_game[0], block_type[0][0], block_type[0][1], block_type[0][2], block_type[0][3],
                      block_type[0][4]],
                     [position[1][0], position[1][1], position[1][2], orientation[1][0], orientation[1][1],
                      orientation[1][2], orientation[1][3], orientation[1][4], size[1][0], color[1][0], color[1][1],
                      color[1][2], in_game[1], block_type[1][0], block_type[1][1], block_type[1][2], block_type[1][3],
                      block_type[1][4]],
                     [position[2][0], position[2][1], position[2][2], orientation[2][0], orientation[2][1],
                      orientation[2][2], orientation[2][3], orientation[2][4], size[2][0], color[2][0], color[2][1],
                      color[2][2], in_game[2], block_type[2][0], block_type[2][1], block_type[2][2], block_type[2][3],
                      block_type[2][4]]]

            next_state = [[next_position[0][0], next_position[0][1], next_position[0][2], next_orientation[0][0],
                           next_orientation[0][1], next_orientation[0][2], next_orientation[0][3],
                           next_orientation[0][4], size[0][0], color[0][0], color[0][1], color[0][2], in_game_next[0],
                           block_type[0][0], block_type[0][1], block_type[0][2], block_type[0][3], block_type[0][4]],
                          [next_position[1][0], next_position[1][1], next_position[1][2], next_orientation[1][0],
                           next_orientation[1][1], next_orientation[1][2], next_orientation[1][3],
                           next_orientation[1][4], size[1][0], color[1][0], color[1][1], color[1][2], in_game_next[1],
                           block_type[1][0], block_type[1][1], block_type[1][2], block_type[1][3], block_type[1][4]],
                          [next_position[2][0], next_position[2][1], next_position[2][2], next_orientation[2][0],
                           next_orientation[2][1], next_orientation[2][2], next_orientation[2][3],
                           next_orientation[2][4], size[2][0], color[2][0], color[2][1], color[2][2], in_game_next[2],
                           block_type[2][0], block_type[2][1], block_type[2][2], block_type[2][3], block_type[2][4]]]
            if j == 0:
                states.append(state)
        states_target.append(next_state)

with open("datasets/test_states_target.json", 'w') as f:
    json.dump(states_target, f, indent=2)

with open("datasets/test_states.json", 'w') as f:
    json.dump(states, f, indent=2)
