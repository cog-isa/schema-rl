import numpy as np
from .constants import Constants
from .graph_utils import Schema, Node, Attribute, Action, Reward
from .tensor_handler import TensorHandler
from .planner import Planner


class SchemaNetwork(Constants):
    def __init__(self, W, R, proxy_env):
        """
        :param W: list of M matrices, each of [(MR + A) x L] shape
        :param R: list of 2 matrices, each of [(MN + A) x L] shape, 1st - pos, 2nd - neg
        """
        # assert matrices are boolean
        for w in W:
            assert (w.dtype == bool)
            if w.ndim != 2:
                print(w.ndim)
                print(w)
                raise AssertionError
            assert (w.shape[0] == self.M * (self.NEIGHBORS_NUM + 1) + self.ACTION_SPACE_DIM)

            assert (w.shape[1] != 0)
        for r in R:
            assert (r.dtype == bool)

        self._W = W
        self._R = R

        self._attribute_nodes = None  # tensor (T+1 x N x M)
        self._action_nodes = None  # tensor (T x ACTION_SPACE_DIM)
        self._reward_nodes = None  # tensor (T x REWARD_SPACE_DIM)

        self._gen_attribute_nodes()
        self._gen_action_nodes()
        self._gen_reward_nodes()

        self._tensor_handler = TensorHandler(self._W, self._R, self._attribute_nodes,
                                             self._action_nodes, self._reward_nodes,
                                             proxy_env)

        self._planner = Planner(self._reward_nodes)

    def _gen_attribute_node_matrix(self, t):
        n_rows = self.N
        n_cols = self.M
        is_active = True if t == 0 else False
        matrix = [
            [Attribute(entity_idx, attribute_idx, t, is_active) for attribute_idx in range(n_cols)]
            for entity_idx in range(n_rows)
        ]
        return matrix

    def _gen_attribute_nodes(self):
        tensor = [self._gen_attribute_node_matrix(t) for t in range(self.T + 1)]
        self._attribute_nodes = np.array(tensor)

    def _gen_action_nodes(self):
        action_nodes = [
            [Action(idx, t=t) for idx in range(self.ACTION_SPACE_DIM)]
            for t in range(self.T + 1)
        ]
        self._action_nodes = np.array(action_nodes)

    def _gen_reward_nodes(self):
        reward_nodes = [
            [Reward(idx, t=t) for idx in range(self.REWARD_SPACE_DIM)]
            for t in range(self.T + 1)
        ]
        self._reward_nodes = np.array(reward_nodes)

    def plan_actions(self):
        """
        proxy_env must be set before calling this
        """

        # instantiate schemas, determine nodes feasibility
        self._tensor_handler.forward_pass()

        # planning actions
        actions = self._planner.plan_actions()

        return actions
