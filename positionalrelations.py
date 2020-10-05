class PositionalRelations:
    def __init__(self, shape, objects):
        self.shape = shape
        self.relationto = {}
        self.initialize(objects)

    def initialize(self, objects):
        for object in objects:
            self.relationto[object] = self.get_positional_relation(object)

    def get_positional_relation(self, object):
        pos_a = self.shape.getPosition()
        pos_b = object.getPosition()
        po_rel = []
        po_rel[0] = pos_a[0] - pos_b[0]
        po_rel[1] = pos_a[1] - pos_b[1]
        po_rel[2] = pos_a[2] - pos_b[2]

        return po_rel

    def update_relations(self, objects):
        for object in objects:
            self.relationto[object] = self.get_positional_relation(object)