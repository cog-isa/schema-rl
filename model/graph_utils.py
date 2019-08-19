import numpy as np
from .constants import Constants


class Schema:
    """Grounded schema type."""

    def __init__(self, attribute_preconditions, action_preconditions):
        """
        preconditions: list of Nodes
        """
        self.attribute_preconditions = attribute_preconditions
        self.action_preconditions = action_preconditions
        self.is_reachable = None
        self.ancestor_actions = None


class Node:
    def __init__(self):
        """
        value: bool variable
        schemas: list of schemas
        is_discovered: has node been seen during graph traversal
        """
        self.is_feasible = False

        self.is_reachable = None
        self.reachable_by = None  # reachable by this schema
        self.value = None
        self.schemas = []
        self.ancestor_actions = None

    def add_schema(self, preconditions):
        # in current implementation schemas are instantiated only on feasible nodes
        if not self.is_feasible:
            self.is_feasible = True

        attribute_preconditions = []
        action_preconditions = []

        for precondition in preconditions:
            if type(precondition) is Attribute:
                attribute_preconditions.append(precondition)
            elif type(precondition) is Action:
                # assuming only one action precondition
                # it's needed for simple action planning during reward backtrace
                if len(action_preconditions) >= 1:
                    print('schema is preconditioned more than on one action')
                    raise AssertionError
                action_preconditions.append(precondition)
            else:
                raise AssertionError

        self.schemas.append(
            Schema(attribute_preconditions, action_preconditions)
        )


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
    not_planned_idx = -1

    def __init__(self, idx):
        """
        action_idx: action unique idx
        """
        self.idx = idx


class Reward(Node):
    pos_idx = 0
    allowed_signs = ('pos', 'neg')

    def __init__(self, idx):
        self.idx = idx
        self.sign = self.allowed_signs[idx]
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



















