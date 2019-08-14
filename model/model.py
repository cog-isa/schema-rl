import numpy as np
from .graph_utils import *

from .featurematrix import FeatureMatrix


class SchemaNetwork:
    def __init__(self, N, M, A, L, T):
        """
        N: number of entities
        M: number of attributes of each entity
        A: number of available actions
        L: number of schemas
        T: size of look-ahead window
        required_actions: actions agent should take to reach planned reward
        """
        self._N = N
        self._M = M
        self._A = A
        self._L = L
        self._T = T

        # list of M matrices, each of [(MR + A) x L] shape
        self._W = []

        # list of 2 matrices, each of [(MN + A) x L] shape
        # first - positive reward
        # second - negative reward
        self._R = []

        self._proxy_env = None
        self._attribute_tensor = None
        self._attribute_graph = None

    def set_proxy_env(self, proxy_env):
        self._proxy_env = proxy_env

    def _gen_attribute_tensor(self):
        shape = (self._N, self._M, self._T)
        self._attribute_tensor = np.empty(shape, dtype=bool)

    def _gen_attribute_matrix_deprecated(self):
        n_rows = self._N
        n_cols = self._M
        matrix = [
            [Attribute(entity_idx, attribute_idx) for attribute_idx in range(n_cols)]
            for entity_idx in range(n_rows)]
        return matrix

    def _gen_attribute_graph(self):
        n_times = self._T
        tensor = [self._gen_attribute_matrix() for t in range(n_times)]
        #self._attribute_tensor = tensor

    def _get_env_attribute_matrix(self):
        """
        Get observed state at t = 0 as (N x M) matrix
        """
        attribute_matrix = self._proxy_env.get_attribute_matrix()
        return attribute_matrix

    def _convert_matrix(self, X):
        """
        Convert (N x M) matrix to (N x (MR + A)) matrix
        """
        converted_matrix= self._proxy_env.transform_matrix(custom_matrix=X,
                                                           add_all_actions=True)
        return converted_matrix

    def _init_attribute_tensor(self, X):
        """
        :param X: (N x M) ndarray of attributes at time t = 0
        """
        time = 0
        self._attribute_tensor[:, :, time] = X.copy()

    def _init_attribute_layer(self, X):
        """
        :param X: (N x M) ndarray of attributes at time t = 0
        """
        time = 0
        self._attribute_tensor[:, :, time] = X.copy()
        n_rows = self._N
        n_cols = self._M
        for i in range(n_rows):
            for j in range(n_cols):
                self._attribute_tensor[i][j][time].value = X[i, j]

    def _predict_next_attributes(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        # check dimensions
        #assert (X.shape == (self._N, (self._M * self._R + self._A)))
        #for W in self._W:
            #assert (W.shape[0] == (self._M * self._R + self._A))
            #assert (W.dtype == bool)

        X = self._attribute_tensor[:, :, t]
        next_X = self._attribute_tensor[:, :, t+1]

        for idx, W in enumerate(self._W):
            prediction_matrix = ~(~X @ W)
            # TODO: set schemas here
            next_X[:idx] = prediction_matrix.any(axis=1)

    def _predict_next_rewards(self, X):
        """
        X: vector of all attributes and actions, [1 x (NM + A)]
        """
        # check dimensions
        assert (X.shape == (1, (self._N * self._M + self._A)))
        for R in self._R:
            assert (R.shape == ((self._N * self._M + self._A), self._L))
            assert (R.dtype == bool)

        # expecting 2 rewards: pos and neg
        predictions = []
        for R in self._R:
            prediction = ~(~X @ R)
            predictions.append(prediction)

        return tuple(predictions)

    def _forward_pass(self, X, V):
        """
        X: matrix [N x (MR + A)]
        V: matrix [1 x (MN + A)]
        """
        self._gen_attribute_tensor()

        attribute_matrix = self._get_env_attribute_matrix()
        self._init_attribute_tensor(attribute_matrix)

        for t in range(self._T):
            # compute (N x M) matrix of next attributes
            self._predict_next_attributes(t)
            self._predict_next_rewards(t)
