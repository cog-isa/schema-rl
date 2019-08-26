import numpy as np
from .constants import Constants
from .graph_utils import MetaObject, Attribute, FakeAttribute, Action


class TensorHandler(Constants):
    def __init__(self, W, R, attribute_nodes, action_nodes, reward_nodes):
        self._W = W
        self._R = R

        self._proxy_env = None
        self._attribute_tensor = None
        self._reward_tensor = None

        # from SchemaNetwork
        self._attribute_nodes = attribute_nodes
        self._action_nodes = action_nodes
        self._reward_nodes = reward_nodes

        # create tensors for storing state
        self._gen_attribute_tensor()
        self._gen_reward_tensor()
        
    def set_proxy_env(self, env):
        self._proxy_env = env

    def _get_env_attribute_matrix(self):
        """
        Get observed state at t = 0
        :returns (N x M) matrix
        """
        assert (self._proxy_env is not None)
        # attribute_matrix = self._proxy_env.get_attribute_matrix()
        print('STUB: get_env_attribute_matrix()')

        attribute_matrix = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
        attribute_matrix = np.reshape(attribute_matrix, (9, 1))
        attribute_matrix = attribute_matrix.astype(bool)

        return attribute_matrix

    def _gen_attribute_tensor(self):
        shape = (self.T + 1, self.N, self.M)
        self._attribute_tensor = np.empty(shape, dtype=bool)

    def _gen_reward_tensor(self):
        shape = (self.T + 1, self.REWARD_SPACE_DIM)
        self._reward_tensor = np.empty(shape, dtype=bool)

    def _make_reference_matrix(self, metadata_matrix, t):
        """
        :param metadata_matrix: (N x (MR + A)) or (1 x (NM + A)) matrix of MetaObject's
        :param t: layer of tensor to which references are established
        :return: (N x (MR + A)) or (1 x (NM + A)) matrix of references to nodes
        """
        n_rows, m_cols = metadata_matrix.shape
        reference_matrix = np.empty((n_rows, m_cols), dtype=object)
        for i in range(n_rows):
            for j in range(m_cols):
                meta_object = metadata_matrix[i, j]
                if meta_object.obj_type is Attribute:
                    reference = self._attribute_nodes[t, meta_object.entity_idx, meta_object.attribute_idx]
                elif meta_object.obj_type is FakeAttribute:
                    reference = None
                elif meta_object.obj_type is Action:
                    reference = self._action_nodes[t, meta_object.action_idx]
                else:
                    raise AssertionError
                reference_matrix[i, j] = reference
        return reference_matrix

    def _transform_matrix(self, matrix, t, output_format):
        """
        for 'attribute: convert (N x M) to (N x (MR + A))
        for 'reward': convert (N x M) to (1 x (MN + A))
        :param t: time step where we got matrix
        """
        transformed_matrix, metadata_matrix = \
            self._proxy_env.transform_matrix(matrix=matrix,
                                             output_format=output_format)
        reference_matrix = self._make_reference_matrix(metadata_matrix, t)
        return transformed_matrix, reference_matrix

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
        transformed_matrix, reference_matrix = self._transform_matrix(src_matrix, t, output_format='attribute')

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
        transformed_matrix, reference_matrix = self._transform_matrix(src_matrix, t, output_format='reward')

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
