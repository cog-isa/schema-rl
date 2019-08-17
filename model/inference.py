import numpy as np
from .constants import Constants
from .graph_utils import Schema, Node, Attribute, Action, Reward
from .tensor_handler import TensorHandler
from .planner import Planner


class SchemaNetwork(Constants):
    def __init__(self, W, R):
        """
        :param W: list of M matrices, each of [(MR + A) x L] shape
        :param R: list of 2 matrices, each of [(MN + A) x L] shape, 1st - pos, 2nd - neg
        """
        self._W = W
        self._R = R

        self._attribute_nodes = None  # tensor (N x M x T)
        self._action_nodes = None  # tensor (T x ACTION_SPACE_DIM)
        self._reward_nodes = None  # tensor (T x REWARD_SPACE_DIM)

        self._tensor_handler = TensorHandler(self._W, self._R, self._attribute_nodes,
                                             self._action_nodes, self._reward_nodes)
        self._planner = Planner()

    def set_proxy_env(self, env):
        self._tensor_handler.set_proxy_env(env)

    def _gen_attribute_node_matrix(self):
        n_rows = self.N
        n_cols = self.M
        matrix = [
            [Attribute(entity_idx, attribute_idx) for attribute_idx in range(n_cols)]
            for entity_idx in range(n_rows)]
        return matrix

    def _gen_attribute_nodes(self):
        n_times = self.T
        tensor = [self._gen_attribute_node_matrix() for t in range(n_times)]
        self._attribute_nodes = np.array(tensor)

    def _gen_action_nodes(self):
        action_nodes = [[Action(0) for dim in range(self.ACTION_SPACE_DIM)]
                        for t in range(self.T)]
        self._action_nodes = np.array(action_nodes)

    def _set_node_layer(self, t):
        """
        t: time step
        """
        n_rows = self.N
        n_cols = self.M
        for i in range(n_rows):
            for j in range(n_cols):
                self._attribute_nodes[i][j][t].value = self._attribute_tensor[i, j, t]

    def _forward_pass(self):
        """
        X: matrix [N x (MR + A)]
        V: matrix [1 x (MN + A)]
        """
        self._gen_attribute_nodes()
        self._gen_action_nodes()

        # set connections of attribute_nodes
        self._tensor_handler.forward_pass()

    def _plan_actions(self):
        self._forward_pass()

        # find closest positive reward
