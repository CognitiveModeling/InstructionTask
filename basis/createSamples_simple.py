import numpy as np
import json
import random
import torch
import math


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


r = list(range(0, 4600))
# random.shuffle(r)

n_agents = 3
n_actions = 5
n_positions = 3
n_state_positions = 4
n_orientations = 6
counter = 0
states = []
states_target = []
agents = []
positions = []

for i in r:
    old_position = [[], [], []]
    position = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    next_position = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    action = [0, 0, 0]
    reference_block = -1
    orig_ref_block = -1
    old_next_position = [[], [], []]
    orientation = [[], [], []]
    next_orientation = [[], [], []]
    with open('arrangements/arrangement' + str(i) + '.json') as json_file:
        data = json.load(json_file)

        for j in range(n_actions):
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
                if data[j][l][3][2] != -1:
                    in_game.append(1)
                else:
                    in_game.append(0)
                if data[j + 1][l][3][2] != -1:
                    in_game_next.append(1)
                else:
                    in_game_next.append(0)
                color.append(data[j][l][1])
                size.append(data[j][l][2])
                if len(data[j][l]) == 6:
                    old_position[l] = data[j][l][3]
                    orientation[l] = data[j][l][4]
                else:
                    orientation[l] = data[j][l][3]
                if len(data[j + 1][l]) == 6:
                    old_next_position[l] = data[j + 1][l][3]
                    next_orientation[l] = data[j + 1][l][4]
                else:
                    next_orientation[l] = data[j + 1][l][3]
                    old_next_position[l] = old_position[l]
                if is_orig_ref_block(old_position[l]):
                    orig_ref_block = l

            if orig_ref_block == -1:
                for l in range(n_agents):
                    if is_orig_ref_block(old_next_position[l]):
                        next_ref_block = l
            else:
                next_ref_block = orig_ref_block
            block_to_be_moved = data[j][n_agents][0][0]
            target_position = data[j][n_agents][1]
            target_orientation = data[j][n_agents][2]
            for m in range(n_agents):
                all_blocks = list(range(n_agents))
                all_blocks.remove(m)
                a = all_blocks[0]
                b = all_blocks[1]
                c1 = old_position[m][0]
                c2 = old_position[m][1]
                if in_game[a] == 1:
                    a1 = old_position[a][0]
                    a2 = old_position[a][1]
                else:
                    a1 = c1
                    a2 = c2
                if in_game[b] == 1:
                    b1 = old_position[b][0]
                    b2 = old_position[b][1]
                else:
                    b1 = c1
                    b2 = c2
                # current position
                if in_game[m] == 0:
                    position[m] = [1, 0, 0, 1, 0, 0]
                elif old_position[m][2] == 0:
                    position[m] = [0, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                elif old_position[m][3] == 0:
                    if in_game[a] == 0:
                        position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]
                    elif in_game[b] == 0:
                        position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                    elif old_position[a][2] == 0 and old_position[b][2] == 0:
                        if point_distance([a1, a2], [c1, c2]) < point_distance([b1, b2], [c1, c2]):
                            position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                        else:
                            position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]
                    elif old_position[a][2] == 1:
                        position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]
                    else:
                        position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                elif old_position[a][2] == 1:
                    position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                else:
                    position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]

                # next position
                c1 = old_next_position[m][0]
                c2 = old_next_position[m][1]
                if in_game_next[a] == 1:
                    a1 = old_next_position[a][0]
                    a2 = old_next_position[a][1]
                else:
                    a1 = c1
                    a2 = c2
                if in_game_next[b] == 1:
                    b1 = old_next_position[b][0]
                    b2 = old_next_position[b][1]
                else:
                    b1 = c1
                    b2 = c2
                if in_game_next[m] == 0:
                    next_position[m] = [1, 0, 0, 1, 0, 0]
                elif old_next_position[m][2] == 0:
                    next_position[m] = [0, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                elif old_next_position[m][3] == 0:
                    if in_game_next[a] == 0:
                        next_position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]
                    elif in_game_next[b] == 0:
                        next_position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                    elif old_next_position[a][2] == 0 and old_next_position[b][2] == 0:
                        if point_distance([a1, a2], [c1, c2]) < point_distance([b1, b2], [c1, c2]):
                            next_position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                        else:
                            next_position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]
                    elif old_next_position[a][2] == 1:
                        next_position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]
                    else:
                        next_position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                elif old_next_position[a][2] == 1:
                    next_position[m] = [1, c1 - a1, c2 - a2, 0, c1 - b1, c2 - b2]
                else:
                    next_position[m] = [0, c1 - a1, c2 - a2, 1, c1 - b1, c2 - b2]

            # action
            reference_block = orig_ref_block
            all_blocks = list(range(n_agents))
            all_blocks.remove(block_to_be_moved)
            if all_blocks[0] == reference_block:
                other_block = all_blocks[1]
            else:
                other_block = all_blocks[0]
            if target_position[2] < 0:
                action = [-1, target_position[0], target_position[1]]
            elif old_next_position[block_to_be_moved][3] == 1:
                if all_blocks[0] == orig_ref_block:
                    reference_block = all_blocks[1]
                else:
                    reference_block = all_blocks[0]
                action = [1, target_position[0] - old_position[reference_block][0], target_position[1] -
                          old_position[reference_block][1]]
            elif old_next_position[block_to_be_moved][2] == 1:
                if in_game[other_block] == 0:
                    action = [1, target_position[0], target_position[1]]
                elif point_distance([target_position[0], target_position[1]], [old_position[reference_block][0],
                                                                               old_position[reference_block][
                                                                                   1]]) <= point_distance(
                    [target_position[0], target_position[1]],
                    [old_position[other_block][0], old_position[other_block][1]]):
                    action = [1, target_position[0], target_position[1]]
                else:
                    reference_block = other_block
                    action = [1, target_position[0] - old_position[reference_block][0],
                              target_position[1] - old_position[reference_block][1]]
            elif old_next_position[block_to_be_moved][2] == 0:
                action = [0, target_position[0], target_position[1]]
            else:  # placement did not work
                if target_position[2] <= 0.1:
                    action = [0, target_position[0], target_position[1]]
                else:
                    d1 = point_distance([target_position[0], target_position[1]],
                                        [old_position[reference_block][0], old_position[reference_block][1]])
                    if in_game[other_block] == 1:
                        d2 = point_distance([target_position[0], target_position[1]],
                                            [old_position[other_block][0], old_position[other_block][1]])
                    else:
                        d2 = 10
                    h1 = size[reference_block][0]
                    h2 = size[other_block][0]
                    z = target_position[2]
                    if z < h1 and z < h2:
                        action = [0, target_position[0], target_position[1]]
                    elif z < h2:
                        action = [1, target_position[0], target_position[1]]
                    elif z < h1:
                        reference_block = other_block
                        action = [1, target_position[0] - old_position[reference_block][0],
                                  target_position[1] - old_position[reference_block][1]]
                    elif d1 < d2:
                        action = [1, target_position[0], target_position[1]]
                    else:
                        reference_block = other_block
                        action = [1, target_position[0] - old_position[reference_block][0],
                                  target_position[1] - old_position[reference_block][1]]

            state = [color[0][0], color[0][1], color[0][2], size[0][0], in_game[0], position[0][0], position[0][1],
                     position[0][2], position[0][3], position[0][4], position[0][5], orientation[0][0],
                     orientation[0][1], orientation[0][2], orientation[0][3], orientation[0][4], orientation[0][5],
                     color[1][0], color[1][1], color[1][2], size[1][0], in_game[1], position[1][0], position[1][1],
                     position[1][2], position[1][3], position[1][4], position[1][5], orientation[1][0],
                     orientation[1][1], orientation[1][2], orientation[1][3], orientation[1][4], orientation[1][5],
                     color[2][0], color[2][1], color[2][2], size[2][0], in_game[2], position[2][0], position[2][1],
                     position[2][2], position[2][3], position[2][4], position[2][5], orientation[2][0],
                     orientation[2][1], orientation[2][2], orientation[2][3], orientation[2][4], orientation[2][5]]
            next_state = [color[0][0], color[0][1], color[0][2], size[0][0], in_game_next[0], next_position[0][0],
                          next_position[0][1],
                          next_position[0][2], next_position[0][3], next_position[0][4], next_position[0][5],
                          next_orientation[0][0],
                          next_orientation[0][1], next_orientation[0][2], next_orientation[0][3],
                          next_orientation[0][4], next_orientation[0][5],
                          color[1][0], color[1][1], color[1][2], size[1][0], in_game_next[1], next_position[1][0],
                          next_position[1][1],
                          next_position[1][2], next_position[1][3], next_position[1][4], next_position[1][5],
                          next_orientation[1][0],
                          next_orientation[1][1], next_orientation[1][2], next_orientation[1][3],
                          next_orientation[1][4], next_orientation[1][5],
                          color[2][0], color[2][1], color[2][2], size[2][0], in_game_next[2], next_position[2][0],
                          next_position[2][1],
                          next_position[2][2], next_position[2][3], next_position[2][4], next_position[2][5],
                          next_orientation[2][0],
                          next_orientation[2][1], next_orientation[2][2], next_orientation[2][3],
                          next_orientation[2][4], next_orientation[2][5]]
            states.append(state)
            states_target.append(next_state)
            agents.append([block_to_be_moved])
            action_pack = [reference_block, action[0], action[1], action[2], target_orientation[0],
                           target_orientation[1], target_orientation[2], target_orientation[3], target_orientation[4],
                           target_orientation[5]]
            positions.append(action_pack)

with open("states.json", 'w') as f:
    json.dump(states, f, indent=2)

with open("states_target.json", 'w') as f:
    json.dump(states_target, f, indent=2)

with open("positions.json", 'w') as f:
    json.dump(positions, f, indent=2)

with open("agents.json", 'w') as f:
    json.dump(agents, f, indent=2)
