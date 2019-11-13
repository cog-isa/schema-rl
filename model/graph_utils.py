import numpy as np
from .constants import Constants


class Schema(Constants):
    """Grounded schema type."""

    def __init__(self, t, attribute_preconditions, action_preconditions):
        """
        preconditions: list of Nodes
        """
        self.t = t
        self.attribute_preconditions = attribute_preconditions
        self.action_preconditions = action_preconditions
        self.is_reachable = None
        self.required_cumulative_actions = None
        self.harmfulness = None

    def compute_cumulative_actions(self):
        """
        index of schema_t is the time_step of its output
        schema_t has exactly len(required_cumulative_actions) == t
        but last elem may not be used (all preconditions are 2 layers down)
        child schemas should have buffer of (t-1) or (t-2) size
        """
        merging_buff = [[] for _ in range(self.t)]

        for attribute_node in self.attribute_preconditions:
            if attribute_node.activating_schema is not None:
                assert (attribute_node.t >= self.FRAME_STACK_SIZE)

                to_merge = attribute_node.activating_schema.required_cumulative_actions

                # e.g. buff has (t-1) size, to_merge has (t-2) size
                assert (self.t - 2) <= len(to_merge) <= (self.t - 1)

                for i in range(len(to_merge)):
                    if len(merging_buff[i]) == 0:
                        merging_buff[i].extend(to_merge[i])
                    else:
                        if merging_buff[i] != to_merge[i]:
                            pass
                            # print('merge conflict detected')
            else:
                assert attribute_node.t < self.FRAME_STACK_SIZE \
                    or attribute_node.attribute_idx == self.VOID_IDX

        # add action preconditions of current schema
        for action_node in self.action_preconditions:
            merging_buff[action_node.t].append(action_node.idx)

        self.required_cumulative_actions = merging_buff

    def _get_margin(self):
        """
        margin only by attributes!
        """
        margin = []
        for attr in self.attribute_preconditions:
            if attr.is_reachable is None:
                margin.append(attr)
        return margin

    def compute_harmfulness(self, neg_schemas):
        relative_harms = []
        for neg_schema in neg_schemas:
            margin = neg_schema._get_margin()
            if len(margin) != 0:
                intersection = list(set(margin) & set(self.attribute_preconditions))
                harm = len(intersection) / len(margin)
            else:
                harm = 0
            relative_harms.append(harm)
        self.harmfulness = max(relative_harms)


class Node:
    def __init__(self, t):
        """
        schemas: list of schemas
        """
        self.t = t

        self.is_feasible = False
        self.is_reachable = None
        self.activating_schema = None  # reachable by this schema
        self.schemas = []

    def reset(self):
        self.is_feasible = False
        self.is_reachable = None
        self.activating_schema = None
        self.schemas.clear()

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
            Schema(self.t, attribute_preconditions, action_preconditions)
        )

    def sort_schemas_by_harmfulness(self, neg_schemas):
        """
        :param neg_schemas: list of schemas for negative reward at the same time step
                            as node's time step
        """
        for schema in self.schemas:
            schema.compute_harmfulness(neg_schemas)

        self.schemas = sorted(self.schemas,
                              key=lambda x: x.harmfulness)


class Attribute(Node, Constants):
    def __init__(self, entity_idx, attribute_idx, t):
        """
        entity_idx: entity unique idx [0, N)
        attribute_idx: attribute index in entity's attribute vector
        """
        self.entity_idx = entity_idx
        self.attribute_idx = attribute_idx
        super().__init__(t)

    def reset(self):
        super().reset()

        if self.t < self.FRAME_STACK_SIZE or self.attribute_idx == self.VOID_IDX:
            self.is_reachable = True


class FakeAttribute:
    """
    Attribute of entity that is out of screen (zero-padding in matrix)
    """
    pass


class Action:
    not_planned_idx = 0

    def __init__(self, idx, t):
        """
        action_idx: action unique idx
        """
        self.idx = idx
        self.t = t


class Reward(Node):
    sign2idx = {'pos': 0,
                'neg': 1}

    allowed_signs = sign2idx.keys()

    def __init__(self, idx, t):
        self.idx = idx
        super().__init__(t)

    def reset(self):
        super().reset()


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



















