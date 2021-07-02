import numpy as np
import json
import random
import torch
import math

r = list(range(4900, 5000))
#random.shuffle(r)

n_agents = 3
n_actions = 3
n_positions = 8
n_orientations = 11

agents = []
positions = []
states = []
counter = 0

for i in r:
    actor = []
    position = []
    state = []
    problematic = False
    with open('arrangements_relative/arrangement' + str(i) + '.json') as json_file:
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
                for m in range(4):
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
                            if data[j][l][m][0] == -1:
                                helper.append(-1)
                            else:
                                helper.append(data[j][l][m+1][n])
                    else:
                        for n in range(3):
                            if math.isnan(data[j][l][m+1][n]):
                                problematic = True
                                break
                            helper.append(data[j][l][m+1][n])

            if math.isnan(data[j][n_agents][0][0]) or math.isnan(data[j][n_agents][1][0]):
                problematic = True

            if problematic:
                break
            state.append(helper)
            actor.append(data[j][n_agents][0])
            for idx in range(n_positions):
                pos_helper.append(data[j][n_agents][1][idx])
            for idx in range(n_orientations):
                pos_helper.append(data[j][n_agents][2][idx])
            position.append(pos_helper)

        helper = []
        #for item in reference:
        #    helper.append(item)
        for l in range(n_agents):
            helper.append(data[n_actions][l][0])
            if data[n_actions][l][3][2] < 0 or math.isnan(data[j][l][4][2]):
                problematic = True
                break
            for m in range(4):
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
                        helper.append(data[n_actions][l][m+1][n])
                else:
                    for n in range(3):
                        if math.isnan(data[n_actions][l][m+1][n]):
                            problematic = True
                            break
                        helper.append(data[n_actions][l][m+1][n])

        state.append(helper)

        if problematic:
            counter += 1
            continue

    agents.append(actor)
    positions.append(position)
    states.append(state)

print(counter)

with open("test_states_relative_additional.json", 'w') as f:
            json.dump(states, f, indent=2)

with open("test_positions_relative_additional.json", 'w') as f:
            json.dump(positions, f, indent=2)

with open("test_agents_relative_additional.json", 'w') as f:
            json.dump(agents, f, indent=2)



