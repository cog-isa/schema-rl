import numpy as np
from .constants import Constants
from .shaper import Shaper
from .graph_utils import MetaObject, Attribute, FakeAttribute, Action


class TensorHandler(Constants):
    def __init__(self, W, R, attribute_nodes, action_nodes, reward_nodes, proxy_env):
        self._W = W
        self._R = R

        self._proxy_env = proxy_env
        self._attribute_tensor = None
        self._reward_tensor = None

        # from SchemaNetwork
        self._attribute_nodes = attribute_nodes
        self._action_nodes = action_nodes
        self._reward_nodes = reward_nodes

        # helping tensor for instantiating schemas
        self._reference_attribute_nodes = None  # tensor ((T+1) x N x (MR + A))

        # shaping matrices and node tensors
        self._shaper = Shaper()

        # create tensors
        self._gen_attribute_tensor()
        self._gen_reward_tensor()
        self._gen_reference_attribute_nodes()

    def _get_env_attribute_matrix(self):
        """
        Get observed state at t = 0
        :returns (N x M) matrix
        """
        if self._proxy_env is None:
            print('set proxy_env before calling planner')
            raise AssertionError

        if self.DEBUG:
            print('STUB: get_env_attribute_matrix()')
            attribute_matrix = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
            attribute_matrix = np.reshape(attribute_matrix, (9, 1))
            attribute_matrix = attribute_matrix.astype(bool)
        else:
            attribute_matrix = self._proxy_env.get_attribute_matrix()

        return attribute_matrix

    def _gen_attribute_tensor(self):
        shape = (self.T + 1, self.N, self.M)
        self._attribute_tensor = np.empty(shape, dtype=bool)

    def _gen_reward_tensor(self):
        shape = (self.T + 1, self.REWARD_SPACE_DIM)
        self._reward_tensor = np.empty(shape, dtype=bool)

    def _gen_reference_attribute_nodes(self):
        # ((T+1) x N x (MR + A))
        shape = (self.T+1, self.N, (self.M * self.NEIGHBORS_NUM + 1 + self.ACTION_SPACE_DIM))
        self._reference_attribute_nodes = np.full(
            shape, None, dtype=object
        )
        for t in range(shape[0]):
            src_matrix = self._attribute_nodes[t, :, :]
            self._reference_attribute_nodes[t, :, :] = self._shaper.transform_node_matrix(
                src_matrix, self._action_nodes, t
            )

    def _get_reference_matrix(self, t):
        """
        :param t: layer of tensor to which references are established,
                  time step where we got matrix
        :return: (N x (MR + A)) matrix of references to nodes
        """
        reference_matrix = self._reference_attribute_nodes[t, :, :]
        return reference_matrix

    def _init_first_attribute_layer(self, matrix):
        """
        :param matrix: (N x M) ndarray of attributes at time t = 0
        """
        time = 0
        self._attribute_tensor[time, :, :] = matrix.copy()

    def _instantiate_attribute_grounded_schemas(self, attribute_idx, t, reference_matrix, W, predicted_matrix):
        """
        :param reference_matrix: (N x (MR + A))
        :param t: schema output time
        """
        for entity_idx in range(self.N):
            activity_mask = predicted_matrix[entity_idx, :]
            precondition_masks = W[:, activity_mask].T

            for mask in precondition_masks:
                preconditions = reference_matrix[entity_idx, mask]
                self._attribute_nodes[t, entity_idx, attribute_idx].add_schema(preconditions)

    def _instantiate_reward_grounded_schemas(self, reward_idx, t, reference_matrix, R, predicted_matrix):
        """
        THIS MAY INSTANTIATE DUPLICATE SCHEMAS!!!
        :param reference_matrix: (1 x (MN + A))
        :param t: schema output time
        """
        """code for old flat rewards:
        activity_mask = np.squeeze(predicted_matrix, axis=0)  # get rid of first dimension
        reference_matrix = np.squeeze(reference_matrix, axis=0)

        precondition_masks = R[:, activity_mask].T

        for mask in precondition_masks:
            preconditions = reference_matrix[mask]
            self._reward_nodes[t, reward_idx].add_schema(preconditions)
        """
        for row_idx in range(self.N):
            activity_mask = predicted_matrix[row_idx, :]
            precondition_masks = R[:, activity_mask].T

            for mask in precondition_masks:
                preconditions = reference_matrix[row_idx, mask]
                self._reward_nodes[t, reward_idx].add_schema(preconditions)

    def _predict_next_attribute_layer(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        src_matrix = self._attribute_tensor[t, :, :]  # get (N x M) matrix
        transformed_matrix = self._shaper.transform_matrix(src_matrix)
        reference_matrix = self._get_reference_matrix(t)

        for attr_idx, W in enumerate(self._W):
            predicted_matrix = ~(~transformed_matrix @ W)
            self._instantiate_attribute_grounded_schemas(attr_idx, t+1, reference_matrix, W, predicted_matrix)
            self._attribute_tensor[t + 1, :, attr_idx] = predicted_matrix.any(axis=1)

    def _predict_next_reward_layer(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        src_matrix = self._attribute_tensor[t, :, :]  # get (N x M) matrix
        transformed_matrix = self._shaper.transform_matrix(src_matrix)
        reference_matrix = self._get_reference_matrix(t)

        for reward_idx, R in enumerate(self._R):
            predicted_matrix = ~(~transformed_matrix @ R)
            self._instantiate_reward_grounded_schemas(reward_idx, t + 1, reference_matrix, R, predicted_matrix)
            self._reward_tensor[t + 1, reward_idx] = predicted_matrix.any()  # OR over all dimensions

    def forward_pass(self):
        """
        Fill attribute_nodes and reward_nodes with schema information
        """
        # init first matrix from env
        attribute_matrix = self._get_env_attribute_matrix()
        self._init_first_attribute_layer(attribute_matrix)

        # propagate forward
        for t in range(self.T):
            self._predict_next_attribute_layer(t)
            self._predict_next_reward_layer(t)
