import numpy as np
import json
import random
import torch
import math


def correct_ref(ref, self):
    if self == 0 and ref != -1:
        ref = ref - 1
    elif self == 1:
        if ref == 2:
            ref = 1
    return ref


def same_orientation(po1, po2):
    if po1[0] == po2[0] and po1[1] == po2[1] and po1[2] == po2[2]:
        return po1[3] - 0.01 <= po2[3] <= po1[3] + 0.01 and po1[4] - 0.01 <= po2[4] <= po1[4] + 0.01


def same_position(po1, po2):
    return po1[0] - 0.01 <= po2[0] <= po1[0] + 0.01 and po1[1] - 0.01 <= po2[1] <= po1[1] + 0.01 and \
           po1[2] - 0.1 <= po2[2] <= po1[2] + 0.1


def same_position_strict(po1, po2):
    return po1[0] == po2[0] and po1[1] == po2[1] and po1[2] == po2[2]


def same_as(v1, v2):
    return v2 - 0.02 < v1 < v2 + 0.02


def is_orig_ref_block(p):
    return same_position_strict(p, [0, 0, 0])


def point_distance(p1, p2):
    return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))


r = list(range(0, 1760))
# random.shuffle(r)

n_agents = 3
n_actions = 5
n_positions = 4
n_state_positions = 6
n_orientations = 6
counter = 0
states = []
states_target = []
agents = []
positions = []

for i in r:
    with open('arrangements_basic/arrangement' + str(i) + '.json') as json_file:
        data = json.load(json_file)

        for j in range(n_actions):
            position = []
            next_position = []
            action = []
            reference_block = -1
            orientation = []
            next_orientation = []
            color = []
            size = []
            in_game = []
            in_game_next = []

            # get all information for all blocks
            for l in range(n_agents):
                # actor = l
                state = []
                state_target = []
                all_blocks = list(range(n_agents))
                all_blocks.remove(l)
                # is block l in the game yet?
                in_game.append(data[j][l][2])
                in_game_next.append(data[j + 1][l][2])
                color.append(data[j][l][0])
                size.append(data[j][l][1])
                if len(data[j][l]) == 5:
                    position.append(data[j][l][3])
                    orientation.append(data[j][l][4])
                else:
                    position.append([1, 0, 0, 1, 0, 0])
                    orientation.append(data[j][l][3])
                if len(data[j + 1][l]) == 5:
                    next_position.append(data[j + 1][l][3])
                    next_orientation.append(data[j + 1][l][4])
                else:
                    next_position.append([1, 0, 0, 1, 0, 0])
                    next_orientation.append(data[j + 1][l][3])
            block_to_be_moved = data[j][n_agents][0]
            if data[j][n_agents][1][0] == -1:
                reference_block = [-1, -1, -1]
            else:
                reference_block = [0, 0, 0]
                reference_block[data[j][n_agents][1][0]] = 1
            action = data[j][n_agents][2]
            target_orientation = data[j][n_agents][3]

            state = [color[0][0], color[0][1], color[0][2], size[0][0], in_game[0][0], position[0][0], position[0][1],
                     position[0][2], position[0][3], position[0][4], position[0][5], orientation[0][0],
                     orientation[0][1], orientation[0][2], orientation[0][3], orientation[0][4], orientation[0][5],
                     color[1][0], color[1][1], color[1][2], size[1][0], in_game[1][0], position[1][0], position[1][1],
                     position[1][2], position[1][3], position[1][4], position[1][5], orientation[1][0],
                     orientation[1][1], orientation[1][2], orientation[1][3], orientation[1][4], orientation[1][5],
                     color[2][0], color[2][1], color[2][2], size[2][0], in_game[2][0], position[2][0], position[2][1],
                     position[2][2], position[2][3], position[2][4], position[2][5], orientation[2][0],
                     orientation[2][1], orientation[2][2], orientation[2][3], orientation[2][4], orientation[2][5]]
            next_state = [color[0][0], color[0][1], color[0][2], size[0][0], in_game_next[0][0], next_position[0][0],
                          next_position[0][1], next_position[0][2], next_position[0][3], next_position[0][4],
                          next_position[0][5],
                          next_orientation[0][0], next_orientation[0][1], next_orientation[0][2],
                          next_orientation[0][3],
                          next_orientation[0][4], next_orientation[0][5],
                          color[1][0], color[1][1], color[1][2], size[1][0], in_game_next[1][0], next_position[1][0],
                          next_position[1][1],
                          next_position[1][2], next_position[1][3], next_position[1][4], next_position[1][5],
                          next_orientation[1][0],
                          next_orientation[1][1], next_orientation[1][2], next_orientation[1][3],
                          next_orientation[1][4], next_orientation[1][5],
                          color[2][0], color[2][1], color[2][2], size[2][0], in_game_next[2][0], next_position[2][0],
                          next_position[2][1],
                          next_position[2][2], next_position[2][3], next_position[2][4], next_position[2][5],
                          next_orientation[2][0],
                          next_orientation[2][1], next_orientation[2][2], next_orientation[2][3],
                          next_orientation[2][4], next_orientation[2][5]]
            states.append(state)
            states_target.append(next_state)
            agents.append(block_to_be_moved)
            action_pack = [reference_block[0], reference_block[1], reference_block[2], action[0], action[1], action[2],
                           target_orientation[0], target_orientation[1], target_orientation[2], target_orientation[3],
                           target_orientation[4], target_orientation[5]]
            positions.append(action_pack)

with open("states.json", 'w') as f:
    json.dump(states, f, indent=2)

with open("states_target.json", 'w') as f:
    json.dump(states_target, f, indent=2)

with open("positions.json", 'w') as f:
    json.dump(positions, f, indent=2)

with open("agents.json", 'w') as f:
    json.dump(agents, f, indent=2)
