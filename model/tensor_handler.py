import numpy as np
from .constants import Constants


class TensorHandler(Constants):
    def __init__(self, W, R):
        self._W = W
        self._R = R

        self._proxy_env = None
        self._attribute_tensor = None

    def set_proxy_env(self, proxy_env):
        self._proxy_env = proxy_env

    def get_env_state(self):
        """
        Get observed state at t = 0 as (N x M) matrix
        """
        assert(self._proxy_env is not None)
        attribute_matrix = self._proxy_env.get_attribute_matrix()
        return attribute_matrix

    def gen_attribute_tensor(self):
        shape = (self.N, self.M, self.T)
        self._attribute_tensor = np.empty(shape, dtype=bool)

    def _convert_matrix(self, X):
        """
        Convert (N x M) matrix to (N x (MR + A)) matrix
        """
        converted_matrix = self._proxy_env.transform_matrix(custom_matrix=X,
                                                           add_all_actions=True)
        return converted_matrix

    def init_attribute_tensor(self, X):
        """
        :param X: (N x M) ndarray of attributes at time t = 0
        """
        time = 0
        self._attribute_tensor[:, :, time] = X.copy()

    def predict_next_attributes(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        X = self._attribute_tensor[:, :, t] # get (N x M) matrix
        X = self._convert_matrix(X) # convert it to (N x (MR + A))
        next_X = self._attribute_tensor[:, :, t+1]

        for idx, W in enumerate(self._W):
            prediction_matrix = ~(~X @ W)
            # TODO: set schemas here
            next_X[:idx] = prediction_matrix.any(axis=1)

    def predict_next_rewards(self, X):
        """
        X: vector of all attributes and actions, [1 x (NM + A)]
        """
        # expecting 2 rewards: pos and neg
        predictions = []
        for R in self._R:
            prediction = ~(~X @ R)
            predictions.append(prediction)

        return tuple(predictions)