import mathown
import shapes
import math
import sim


class Relations:
    def __init__(self):
        self.relations = {"next to", "beside", "by the side of", "contiguous to", "by",
                          "adjacent to", "opposite of", "to the right of",
                          "to the left of", "further right than", "further left than",
                          "on top of", "above", "atop", "upon", "over", "on", "higher than",
                          "in front of", "closer than", "behind", "to the rear of",
                          "at the back of", "further away than", "below", "underneath",
                          "beneath", "lower than"}
        self.relationValues = {}

    def getRelationValues(self, relation):
        return self.relationValues[relation]

    def addRelation(self, relation, gaussxr, gaussxl, gaussyf, gaussyb, gausszl, gausszh):
        self.relationValues[relation] = [gaussxr, gaussxl, gaussyb, gaussyf, gausszh, gausszl]
