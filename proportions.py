import torch
import torch.nn as nn
import torch.nn.functional as F


class Proportions:
    def __init__(self, net):
        self.proportion_descriptions = ["tall", "big", "long", "elongate", "wide", "broad", "great", "large", "huge",
                                       "bulky", "flat", "prolate", "high", "small", "short", "tiny", "low", "narrow",
                                       "slim", "slender"]
        self.net = net


    def get_proportion_values(self, object, description):
        description_id = self.get_description_id(description)
        return self.net(object)[description_id]

    def get_description_id(self, description):
        return self.proportion_descriptions.index(description)