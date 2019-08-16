import numpy as np
from .constants import Constants


class Schema:
    """Grounded schema type."""

    def __init__(self, attribute_preconditions, action_preconditions):
        """
        preconditions: list of Nodes
        """
        self.is_reachable = None
        self.attribute_preconditions = attribute_preconditions
        self.action_preconditions = action_preconditions
        self.ancestor_actions = None


class Node:
    def __init__(self):
        """
        value: bool variable
        schemas: list of schemas
        is_discovered: has node been seen during graph traversal
        """
        # self.is_feasible = None
        self.is_reachable = None
        self.reachable_by = None  # reachable by this schema
        self.value = None
        self.schemas = []
        self.ancestor_actions = None

    def add_schemas(self, schemas_preconditions, attribute_nodes, t, action_nodes):
        """
        list of lists
        """
        # attributes
        for single_schema_preconditions in schemas_preconditions[:3]:
            attribute_preconditions = []
            action_preconditions = []
            for precondition in single_schema_preconditions:
                if type(precondition) is Attribute:
                    i = precondition.entity_idx
                    j = precondition.attribute_idx
                    attribute_preconditions.append(attribute_nodes[i, j, t])
                elif type(precondition) is Action:
                    action_preconditions.append(action_nodes[t, precondition.idx])
                else:
                    assert False
            self.schemas.append(
                Schema(attribute_preconditions, np.array(action_preconditions))
            )

    def get_ancestors(self):
        """
        returns: [L x n_schema_pins] matrix of references to Nodes
        """
        assert (self.schemas is not None)
        return [schema.preconditions for schema in self.schemas]


class Attribute(Node):
    def __init__(self, entity_idx, attribute_idx, global_idx=None):
        """
        entity_idx: entity unique idx [0, N)
        attribute_idx: attribute index in entity's attribute vector
        """
        self.entity_idx = entity_idx
        self.attribute_idx = attribute_idx
        self.global_idx = global_idx
        super().__init__()


class FakeAttribute:
    """
    Attribute of entity that is out of screen (zero-padding in matrix)
    """
    pass


class Action:
    def __init__(self, idx):
        """
        action_idx: action unique idx
        """
        self.idx = idx


class Reward(Node):
    def __init__(self, sign):
        assert (sign in ('pos', 'neg'))
        self.sign = sign
        super().__init__()


class MetaObject:
    _allowed_types = (Attribute, FakeAttribute, Action, Reward)

    def __init__(self, obj_type, entity_idx=None, attribute_idx=None, action_idx=None, reward_idx=None):
        assert (obj_type in self._allowed_types)
        self.obj_type = obj_type
        self.entity_idx = entity_idx
        self.attribute_idx = attribute_idx
        self.action_idx = action_idx
        self.reward_idx = reward_idx


class MetaFactory(Constants):
    def __init__(self):
        self._meta_fake_attribute = MetaObject(FakeAttribute)
        self._meta_actions = [MetaObject(Action, action_idx=action_idx)
                              for action_idx in range(self.ACTION_SPACE_DIM)]

    def gen_meta_entity(self, entity_idx, fake=False):
        """
        :param entity_idx: [0, N)
        :param fake: return fake entity
        :return: list of meta-attributes
        """
        if fake:
            entity = [
                self._meta_fake_attribute for _ in range(self.M)
            ]
        else:
            entity = [
                MetaObject(Attribute, entity_idx=entity_idx, attribute_idx=attr_idx)
                for attr_idx in range(self.M)
            ]
        return entity

    def gen_meta_actions(self):
        return self._meta_actions



















