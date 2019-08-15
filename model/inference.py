import numpy as np
from .constants import Constants
from .graph_utils import *

from .featurematrix import FeatureMatrix


class SchemaNetwork(Constants):
    def __init__(self):
        # list of M matrices, each of [(MR + A) x L] shape
        self._W = []

        # list of 2 matrices, each of [(MN + A) x L] shape
        # first - positive reward
        # second - negative reward
        self._R = []

        self._proxy_env = None
        self._attribute_nodes = None  # tensor (N x M x T)
        self._action_nodes = None  # tensor (T x ACTION_SPACE_DIM)

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

    def _forward_pass(self, X, V):
        """
        X: matrix [N x (MR + A)]
        V: matrix [1 x (MN + A)]
        """
        self._gen_attribute_nodes()
        self._gen_actions_nodes()

        attribute_matrix = self._get_env_attribute_matrix()
        self._init_attribute_tensor(attribute_matrix)

        for t in range(self._T):
            # compute (N x M) matrix of next attributes
            self._predict_next_attributes(t)
            #self._predict_next_rewards(t)
