import numpy as np
import json
import random
import torch
import math


def same_orientation(po1, po2):
    if po1[0] == po2[0] and po1[1] == po2[1] and po1[2] == po2[2] :
        return po1[3] - 0.01 <= po2[3] <= po1[3] + 0.01 and po1[4] - 0.01 <= po2[4] <= po1[4] + 0.01


def same_position(po1, po2):
    return po1[0] - 0.01 <= po2[0] <= po1[0] + 0.01 and po1[1] - 0.01 <= po2[1] <= po1[1] + 0.01 and \
           po1[2] - 0.1 <= po2[2] <= po1[2] + 0.1


r = list(range(0, 2400))
#random.shuffle(r)

n_agents = 3
n_actions = 5
n_positions = 3
n_orientations = 6
orientation_fixed_idxs = list([0, 1, 2, 4])
n_distances = 2

agents = []
positions = []
states = []
states_target = []
counter = 0

for i in r:
    actor = []
    position = []
    state = []
    state_target = []
    problematic = False
    with open('../arrangements/arrangement' + str(i) + '.json') as json_file:
        data = json.load(json_file)

        #first_block = data[0][n_agents][0][0]
        #reference = [0, 0, 0]
        #reference[first_block] = 1

        for j in range(n_actions):
            pos_helper = []
            helper = []
            #for item in reference:
            #    helper.append(item)

            for l in range(n_agents):
                helper.append(data[j][l][0])
                for m in range(5):
                    if m+1 == 3:
                        for n in range(n_positions):
                            if math.isnan(data[j][l][m+1][n]):
                                problematic = True
                                break
                            else:
                                helper.append(data[j][l][m+1][n])
                    elif m+1 ==4:
                        for n in range(n_orientations):
                            if math.isnan(data[j][l][m+1][n]):
                                problematic = True
                                break
                            else:
                                helper.append(data[j][l][m+1][n])
                    elif m+1 ==2:
                        if math.isnan(data[j][l][m + 1][0]):
                            problematic = True
                            break
                        else:
                            helper.append(data[j][l][m + 1][0])
                    elif m+1 == 5:
                        for n in range(n_distances):
                            if math.isnan(data[j][l][m+1][n]):
                                problematic = True
                                break
                            elif data[j][l][m + 1][n] == 10:
                                helper.append(0)
                                #problematic = True
                                #break
                            elif data[j][l][m][0] == -1:
                                helper.append(-1)
                            else:
                                helper.append(data[j][l][m+1][n])
                    else:
                        for n in range(3):
                            if math.isnan(data[j][l][m+1][n]):
                                problematic = True
                                break
                            else:
                                helper.append(data[j][l][m+1][n])

            if math.isnan(data[j][n_agents][0][0]) or math.isnan(data[j][n_agents][1][0]):
                problematic = True

            if problematic:
                break

            block = data[j][n_agents][0]
            state.append(helper)
            if j > 0:
                target_helper = helper.copy()
            actor.append(block)
            for idx in range(n_positions):
                pos_helper.append(data[j][n_agents][1][idx])
            for idx in range(n_orientations):
                pos_helper.append(data[j][n_agents][2][idx])
            position.append(pos_helper)

            if j > 0:
                if same_orientation(data[j-1][n_agents][2], data[j][data[j-1][n_agents][0][0]][4]) and \
                        same_position(data[j-1][n_agents][1], data[j][data[j-1][n_agents][0][0]][3]):
                    target_helper.append(1)
                else:
                    target_helper.append(0)

                state_target.append(target_helper)

        helper = []
        target_helper = []
        #for item in reference:
        #    helper.append(item)
        for l in range(n_agents):
            helper.append(data[n_actions][l][0])
            #if data[n_actions][l][3][2] < 0 or math.isnan(data[j][l][4][2]):
            #    problematic = True
            #    break
            for m in range(5):
                if m+1 == 3:
                    for n in range(n_positions):
                        if math.isnan(data[n_actions][l][m+1][n]):
                            problematic = True
                            break
                        else:
                            helper.append(data[n_actions][l][m+1][n])
                elif m+1 ==4:
                    for n in range(n_orientations):
                        if math.isnan(data[n_actions][l][m+1][n]):
                            problematic = True
                            break
                        else:
                            helper.append(data[n_actions][l][m + 1][n])
                elif m+1 ==5:
                    for n in range(n_distances):
                        if math.isnan(data[n_actions][l][m+1][n]):
                            problematic = True
                            break
                        if data[n_actions][l][m+1][n] == 10:
                            helper.append(0)
                            #problematic = True
                            #break
                        else:
                            helper.append(data[n_actions][l][m+1][n])
                elif m + 1 == 2:
                    if math.isnan(data[n_actions][l][m + 1][0]):
                        problematic = True
                        break
                    helper.append(data[n_actions][l][m + 1][0])
                else:
                    for n in range(3):
                        if math.isnan(data[n_actions][l][m+1][n]):
                            problematic = True
                            break
                        helper.append(data[n_actions][l][m+1][n])

        state.append(helper)
        target_helper = helper.copy()

        if same_orientation(data[n_actions - 1][n_agents][2], data[n_actions][data[n_actions - 1][n_agents][0][0]][4]) and \
                same_position(data[n_actions - 1][n_agents][1], data[n_actions][data[n_actions - 1][n_agents][0][0]][3]):
            target_helper.append(1)
        else:
            target_helper.append(0)

        state_target.append(target_helper)

        if problematic:
            counter += 1
            continue

    agents.append(actor)
    positions.append(position)
    states.append(state)
    states_target.append(state_target)

print(counter)

with open("states.json", 'w') as f:
            json.dump(states, f, indent=2)

with open("states_target.json", 'w') as f:
            json.dump(states_target, f, indent=2)

with open("positions.json", 'w') as f:
            json.dump(positions, f, indent=2)

with open("agents.json", 'w') as f:
            json.dump(agents, f, indent=2)
