import json


# creates samples out of the arrangements, correcting for differences in format


def same_as(v1, v2):
    return v2 - 0.02 < v1 < v2 + 0.02


r = list(range(0, 7600))
s = list(range(0, 8400))

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
    with open('arrangements/arrangement' + str(i) + '.json') as json_file:
        data = json.load(json_file)

        for j in range(n_actions):
            color = []
            size = []
            in_game = []
            in_game_next = []
            position = []
            next_position = []
            action = []
            orientation = []
            next_orientation = []
            help_position = []
            object_type = []

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
                object_type.append([1, 0, 0, 0, 0])

            block_to_be_moved = data[j][n_agents][0][0]
            target_position = data[j][n_agents][1]
            target_orientation = data[j][n_agents][2]

            state = [[position[0][0], position[0][1], position[0][2], orientation[0][0], orientation[0][1],
                      orientation[0][2], orientation[0][3], orientation[0][4], size[0][0], color[0][0], color[0][1],
                      color[0][2], in_game[0], object_type[0][0], object_type[0][1], object_type[0][2],
                      object_type[0][3], object_type[0][4]],
                     [position[1][0], position[1][1], position[1][2], orientation[1][0], orientation[1][1],
                      orientation[1][2], orientation[1][3], orientation[1][4], size[1][0], color[1][0], color[1][1],
                      color[1][2], in_game[1], object_type[1][0], object_type[1][1], object_type[1][2],
                      object_type[1][3], object_type[1][4]],
                     [position[2][0], position[2][1], position[2][2], orientation[2][0], orientation[2][1],
                      orientation[2][2], orientation[2][3], orientation[2][4], size[2][0], color[2][0], color[2][1],
                      color[2][2], in_game[2], object_type[2][0], object_type[2][1], object_type[2][2],
                      object_type[2][3], object_type[2][4]]]

            next_state = [[next_position[0][0], next_position[0][1], next_position[0][2], next_orientation[0][0],
                           next_orientation[0][1], next_orientation[0][2], next_orientation[0][3],
                           next_orientation[0][4], size[0][0], color[0][0], color[0][1], color[0][2], in_game_next[0],
                           object_type[0][0], object_type[0][1], object_type[0][2], object_type[0][3],
                           object_type[0][4]],
                          [next_position[1][0], next_position[1][1], next_position[1][2], next_orientation[1][0],
                           next_orientation[1][1], next_orientation[1][2], next_orientation[1][3],
                           next_orientation[1][4], size[1][0], color[1][0], color[1][1], color[1][2], in_game_next[1],
                           object_type[1][0], object_type[1][1], object_type[1][2], object_type[1][3],
                           object_type[1][4]],
                          [next_position[2][0], next_position[2][1], next_position[2][2], next_orientation[2][0],
                           next_orientation[2][1], next_orientation[2][2], next_orientation[2][3],
                           next_orientation[2][4], size[2][0], color[2][0], color[2][1], color[2][2], in_game_next[2],
                           object_type[2][0], object_type[2][1], object_type[2][2], object_type[2][3],
                           object_type[2][4]]]
            states.append(state)
            states_target.append(next_state)
            agents.append([block_to_be_moved])
            action_pack = [target_position[0], target_position[1], target_position[2], target_orientation[0],
                           target_orientation[1], target_orientation[2], target_orientation[3], target_orientation[4]]
            positions.append(action_pack)

for i in s:
    with open('mixed_arrangements/arrangement' + str(i) + '.json') as json_file:
        data = json.load(json_file)

        for j in range(n_actions):
            color = []
            size = []
            in_game = []
            in_game_next = []
            position = []
            next_position = []
            action = []
            orientation = []
            next_orientation = []
            help_position = []
            object_type = []

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
                object_type.append(type_helper)

            block_to_be_moved = data[j][n_agents][0][0]
            target_position = data[j][n_agents][1]
            target_orientation = data[j][n_agents][2]

            state = [[position[0][0], position[0][1], position[0][2], orientation[0][0], orientation[0][1],
                      orientation[0][2], orientation[0][3], orientation[0][4], size[0][0], color[0][0], color[0][1],
                      color[0][2], in_game[0], object_type[0][0], object_type[0][1], object_type[0][2],
                      object_type[0][3], object_type[0][4]],
                     [position[1][0], position[1][1], position[1][2], orientation[1][0], orientation[1][1],
                      orientation[1][2], orientation[1][3], orientation[1][4], size[1][0], color[1][0], color[1][1],
                      color[1][2], in_game[1], object_type[1][0], object_type[1][1], object_type[1][2],
                      object_type[1][3], object_type[1][4]],
                     [position[2][0], position[2][1], position[2][2], orientation[2][0], orientation[2][1],
                      orientation[2][2], orientation[2][3], orientation[2][4], size[2][0], color[2][0], color[2][1],
                      color[2][2], in_game[2], object_type[2][0], object_type[2][1], object_type[2][2],
                      object_type[2][3], object_type[2][4]]]

            next_state = [[next_position[0][0], next_position[0][1], next_position[0][2], next_orientation[0][0],
                           next_orientation[0][1], next_orientation[0][2], next_orientation[0][3],
                           next_orientation[0][4], size[0][0], color[0][0], color[0][1], color[0][2], in_game_next[0],
                           object_type[0][0], object_type[0][1], object_type[0][2], object_type[0][3],
                           object_type[0][4]],
                          [next_position[1][0], next_position[1][1], next_position[1][2], next_orientation[1][0],
                           next_orientation[1][1], next_orientation[1][2], next_orientation[1][3],
                           next_orientation[1][4], size[1][0], color[1][0], color[1][1], color[1][2], in_game_next[1],
                           object_type[1][0], object_type[1][1], object_type[1][2], object_type[1][3],
                           object_type[1][4]],
                          [next_position[2][0], next_position[2][1], next_position[2][2], next_orientation[2][0],
                           next_orientation[2][1], next_orientation[2][2], next_orientation[2][3],
                           next_orientation[2][4], size[2][0], color[2][0], color[2][1], color[2][2], in_game_next[2],
                           object_type[2][0], object_type[2][1], object_type[2][2], object_type[2][3],
                           object_type[2][4]]]
            states.append(state)
            states_target.append(next_state)
            agents.append([block_to_be_moved])
            action_pack = [target_position[0], target_position[1], target_position[2], target_orientation[0],
                           target_orientation[1], target_orientation[2], target_orientation[3], target_orientation[4]]
            positions.append(action_pack)

with open("datasets/states.json", 'w') as f:
    json.dump(states, f, indent=2)

with open("datasets/states_target.json", 'w') as f:
    json.dump(states_target, f, indent=2)

with open("datasets/positions.json", 'w') as f:
    json.dump(positions, f, indent=2)

with open("datasets/agents.json", 'w') as f:
    json.dump(agents, f, indent=2)
