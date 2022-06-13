from shapes import Shape
import torch
import random


class Instruction:
    def __init__(self, block_num=3):
        self.blocks = list(range(block_num))
        self.attributes = ["tall", "big", "large", "huge", "small", "tiny", "medium-sized", "spherical", "cylindrical",
                           "box-shaped", "cubical", "cone-shaped", "pyramidal", "pointed", "angular", "round", "blue",
                           "blueish", "green", "greenish", "red", "reddish", "yellow", "yellowish", "orange", "purple",
                           "pink", "brown", "brownish", "gray", "white", "black", "light", "dark"]
        self.position_terms = ["at", "on", "next to", "to the left of", "to the right of", "behind", "in front of",
                               "over", "between"]

    def read_instruction(self, instruction):
        # 4 parts: block info + position + ref block info + ref block 2 info
        block = self.blocks.index(instruction[0])
        position = self.position_terms.index(instruction[1])
        if position == 0:
            ref_2 = None
            ref_1 = None
            coordinates = map(int, instruction[2])
            coordinates = list(coordinates)
        elif position != len(self.position_terms) - 1:
            ref_1 = self.blocks.index(instruction[2])
            ref_2 = None
            coordinates = None
        else:
            ref_1 = self.blocks.index(instruction[2])
            ref_2 = self.blocks.index(instruction[3])
            coordinates = None

        return block, position, ref_1, ref_2, coordinates

    def interpret_position(self, attribute, ref_1=None, ref_2=None, coordinates=None):
        if coordinates is not None:
            position = [coordinates[0], coordinates[1], 0]
        elif ref_2 is not None:
            uncertainty1 = random.uniform(-0.1, 0.1)
            uncertainty2 = random.uniform(-0.1, 0.1)
            p2 = ref_1.get_position_adapted()
            p3 = ref_2.get_position_adapted()

            position = [0, 0, 0]
            position[0] = min(p2[0], p3[0]) + abs(p3[0] - p2[0]) * 0.5 + uncertainty1
            position[1] = min(p2[1], p3[1]) + abs(p3[1] - p2[1]) * 0.5 + uncertainty2

        else:
            uncertainty1 = random.uniform(-0.2, 0.2)
            uncertainty2 = random.uniform(-0.2, 0.2)
            p2 = ref_1.get_position_adapted()
            size = ref_1.get_bounding_box()[0]
            position = [0, 0, 0]

            if attribute == 1:
                position[0] = p2[0] + uncertainty1
                position[1] = p2[1] + uncertainty2
                position[2] = size
            elif attribute == 2:
                sgn = random.choice([-1, 1])
                position[0] = p2[0] + sgn * 0.5 + uncertainty1
                position[1] = p2[1] + uncertainty2
            elif attribute == 3:
                position[0] = p2[0] - 0.5 + uncertainty1
                position[1] = p2[1] + uncertainty2
            elif attribute == 4:
                position[0] = p2[0] + 0.5 + uncertainty1
                position[1] = p2[1] + uncertainty2
            elif attribute == 5:
                position[0] = p2[0] + uncertainty1
                position[1] = p2[1] + 0.5 + uncertainty2
            elif attribute == 6:
                position[0] = p2[0] + uncertainty1
                position[1] = p2[1] - 0.5 + uncertainty2
            elif attribute == 7:
                uncertainty3 = random.uniform(0, 0.4)
                position[0] = p2[0] + uncertainty1
                position[1] = p2[1] + uncertainty2
                position[2] = size + uncertainty3

        return position

    def read_input(self, blocks_in_game):
        instruction = []
        block_list = list(range(3))

        print("Enter number of active block (0, 1, 2)")
        block = input()
        instruction.append(int(block))

        block_list.remove(int(block))

        print("Enter relative position term")
        position = input()
        instruction.append(position)

        if position == "at":
            print("Enter x coordinate")
            x = input()
            print("Enter y coordinate")
            y = input()
            coordinates = [x, y]
            instruction.append(coordinates)

        elif position == "between":
            ref_1 = block_list[0]
            ref_2 = block_list[1]
            instruction.append(ref_1)
            instruction.append(ref_2)

        else:
            print("Enter block number of reference block (Available blocks: " + str(blocks_in_game) + ")")
            ref_1 = input()
            instruction.append(int(ref_1))

        return instruction

    def print_instruction(self, instruction):
        if instruction[1] == "at":
            print(
                "Place block " + str(instruction[0]) + " " + instruction[1] + " position " + str(instruction[2]) + ".")
        elif instruction[1] == "between":
            print("Place block " + str(instruction[0]) + " between block " + str(instruction[2]) + " and block " + str(
                instruction[3]) + ".")
        else:
            print("Place block " + str(instruction[0]) + " " + str(instruction[1]) + " block " + str(
                instruction[2]) + ".")
