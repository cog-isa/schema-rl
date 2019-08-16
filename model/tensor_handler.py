import numpy as np
from .constants import Constants


class TensorHandler(Constants):
    def __init__(self, W, R, attribute_nodes, reward_nodes):
        self._W = W
        self._R = R

        self._proxy_env = None
        self._attribute_tensor = None
        self._reward_tensor = None

        # from SchemaNetwork
        self._attribute_nodes = attribute_nodes
        self._reward_nodes = reward_nodes

    def set_proxy_env(self, env):
        self._proxy_env = env

    def _get_env_state(self):
        """
        Get observed state at t = 0
        :returns (N x M) matrix
        """
        assert (self._proxy_env is not None)
        attribute_matrix = self._proxy_env.get_attribute_matrix()
        return attribute_matrix

    def _gen_attribute_tensor(self):
        shape = (self.T, self.N, self.M)
        self._attribute_tensor = np.empty(shape, dtype=bool)

    def _gen_reward_tensor(self):
        shape = (self.T, self.REWARD_SPACE_DIM)
        self._reward_tensor = np.empty(shape, dtype=bool)

    def _transform_matrix(self, matrix, input_format):
        """
        for 'attribute: convert (N x M) to (N x (MR + A))
        for 'reward': convert (N x M) to (1 x (MN + A))
        """
        assert (input_format in ('attribute', 'reward'))

        if input_format == 'attribute':
            transformed_matrix = self._proxy_env.transform_matrix(custom_matrix=matrix,
                                                                add_all_actions=True)
        else:
            transformed_matrix = None  # implement it in FeatureMatrix
        return transformed_matrix

    def _init_first_attribute_layer(self, matrix):
        """
        :param matrix: (N x M) ndarray of attributes at time t = 0
        """
        time = 0
        self._attribute_tensor[time, :, :] = matrix.copy()

    @staticmethod
    def _find_attribute_active_schemas(self, X_nodes, entity_idx, W, prediction_matrix):
        """
        X_nodes - matrix of objects {Attribute() and Action()}
        """
        schema_mask = prediction_matrix[entity_idx, :]
        precondition_masks = W[:, schema_mask].T

        schemas_preconditions = []
        row_nodes = X_nodes[entity_idx]
        for mask in precondition_masks:
            schemas_preconditions.append(
                [node for node, flag in zip(row_nodes, mask) if flag]
            )
        return schemas_preconditions

    @staticmethod
    def _find_reward_active_schemas(self, X_nodes, R, prediction_matrix):
        """
        :param X_nodes: matrix of objects {Attribute() and Action()} [1 x (MN+A)] shape
        :param prediction_matrix: (1 x L) shape
        :returns schemas: ndarray matrix of precondition objects
        """
        schema_mask = prediction_matrix
        precondition_masks = R[:, schema_mask].T
        schemas = []
        for mask in precondition_masks:
            schemas.append(X_nodes[mask])
        return np.array(schemas, dtype=object)

    def _predict_next_attribute_layer(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        X = self._attribute_tensor[:, :, t]  # get (N x M) matrix
        X, X_nodes = self._convert_matrix(X)  # convert it to (N x (MR + A)) and get X_nodes
        next_X = self._attribute_tensor[:, :, t + 1]

        assert (X_nodes.shape[0] == self.N)

        for attr_idx, W in enumerate(self._W):
            prediction_matrix = ~(~X @ W)
            # set schemas here
            for entity_idx in range(self.N):
                schemas_preconditions = \
                    self._find_attribute_active_schemas(X_nodes, entity_idx, W, prediction_matrix)
                self._attribute_nodes[entity_idx, attr_idx, t+1]\
                    .add_schemas(schemas_preconditions, self._attribute_nodes, t)

            next_X[:attr_idx] = prediction_matrix.any(axis=1)

    def _predict_next_reward_layer(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        X = self._reward_tensor[t, :]  # get (N x M) matrix
        X, X_nodes = self._convert_matrix(X)  # convert it to (N x (MN + A)) and get X_nodes
        next_X = self._reward_tensor[:, :, t + 1]

        #assert (X_nodes.shape[0] == self.N)

        for reward_idx, R in enumerate(self._R):
            prediction_matrix = ~(~X @ R)
            # set schemas here
            schemas = self._find_reward_active_schemas(X_nodes, R, prediction_matrix)
            self._reward_nodes[t + 1, reward_idx] \
                .add_schemas(schemas, self._reward_nodes, t)


            next_X[:attr_idx] = prediction_matrix.any(axis=1)

    def forward_pass(self):
        """
        Fill attribute_nodes with schema information
        X: matrix [N x (MR + A)]
        V: matrix [1 x (MN + A)]
        """
        # create tensors for storing state
        self._gen_attribute_tensor()
        self._gen_reward_tensor()

        # init first matrix from env
        attribute_matrix = self._get_env_state()
        self._init_first_attribute_layer(attribute_matrix)

        # propagate forward
        for t in range(self.T):
            # compute (N x M) matrix of next attributes
            self._predict_next_attribute_layer(t)

            # R not yet implemented
            # self._predict_next_rewards(t)
