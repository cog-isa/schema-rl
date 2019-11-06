import numpy as np
from .constants import Constants
from .shaper import Shaper
from .graph_utils import MetaObject, Attribute, FakeAttribute, Action


class TensorHandler(Constants):
    def __init__(self, W, R, attribute_nodes, action_nodes, reward_nodes, entities_stack):
        self._W = W
        self._R = R

        self._entities_stack = entities_stack

        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
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

    def _get_env_attribute_tensor(self):
        """
        Get observed state
        :returns (FRAME_STACK_SIZE x N x M) tensor
        """
        assert self._entities_stack is not None, 'NO_ENTITIES_STACK'
        assert len(self._entities_stack) == self.FRAME_STACK_SIZE, 'BAD_ENTITIES_STACK'

        if self.DEBUG:
            print('STUB: get_env_attribute_tensor()')
            attribute_tensor = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                         1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            attribute_tensor = np.reshape(attribute_tensor, (2, 9, 2))
            attribute_tensor = attribute_tensor.astype(bool)
        else:
            matrix_shape = (self.N, self.M)
            attribute_tensor = np.empty((self.FRAME_STACK_SIZE,) + matrix_shape, dtype=bool)
            for i in range(self.FRAME_STACK_SIZE):
                matrix = self._entities_stack[i]
                assert matrix.shape == matrix_shape, 'BAD_MATRIX_SHAPE'
                attribute_tensor[i, :, :] = matrix

        return attribute_tensor

    def _gen_attribute_tensor(self):
        shape = (self.FRAME_STACK_SIZE + self.T, self.N, self.M)
        self._attribute_tensor = np.full(shape, False, dtype=bool)

    def _gen_reward_tensor(self):
        shape = (self.FRAME_STACK_SIZE + self.T, self.REWARD_SPACE_DIM)
        self._reward_tensor = np.empty(shape, dtype=bool)

    def _gen_reference_attribute_nodes(self):
        # ((FRAME_STACK_SIZE + T) x N x (MR*ss + A))
        shape = (self.FRAME_STACK_SIZE + self.T,
                 self.N,
                 self.SCHEMA_VEC_SIZE)
        self._reference_attribute_nodes = np.full(
            shape, None, dtype=object
        )
        offset = self.FRAME_STACK_SIZE - 1
        for t in range(offset, offset + self.T + 1):
            src_slice = self._get_tensor_slice(t, 'nodes')
            self._reference_attribute_nodes[t, :, :] = self._shaper.transform_node_matrix(
                src_slice, self._action_nodes, t
            )

    def _get_tensor_slice(self, t, tensor_type):
        """
        t: time at which last layer is located
        size of slice is FRAME_STACK_SIZE in total
        """
        assert tensor_type in ('attributes', 'nodes')
        begin = t - self.FRAME_STACK_SIZE + 1

        # prevent possible shape mismatch downwards the stack
        assert begin >= 0, 'TENSOR_SLICE_BAD_ARGS'

        end = t + 1
        index = np.index_exp[max(0, begin): end]
        if tensor_type == 'attributes':
            slice_ = self._attribute_tensor[index]
        else:
            slice_ = self._attribute_nodes[index]
        return slice_

    def _get_reference_matrix(self, t):
        """
        :param t: rightmost FS's layer to which references are established,
                  time step where we got matrix
        :return: (N x (MR + A)) matrix of references to nodes
        """
        reference_matrix = self._reference_attribute_nodes[t, :, :]
        return reference_matrix

    def _init_attributes(self, tensor):
        """
        :param tensor: (FRAME_STACK_SIZE x N x M) ndarray of attributes
        """
        self._attribute_tensor[:self.FRAME_STACK_SIZE, :, :] = tensor

    def _instantiate_attribute_grounded_schemas(self, attribute_idx, t, reference_matrix, W, predicted_matrix):
        """
        :param reference_matrix: (N x (MR + A))
        :param t: schema output time_step
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
        src_slice = self._get_tensor_slice(t, 'attributes')  # (FRAME_STACK_SIZE x N x M)
        transformed_matrix = self._shaper.transform_matrix(src_slice)
        reference_matrix = self._get_reference_matrix(t)

        for attr_idx, W in enumerate(self._W):
            predicted_matrix = ~(~transformed_matrix @ W)
            self._instantiate_attribute_grounded_schemas(attr_idx, t+1, reference_matrix, W, predicted_matrix)
            self._attribute_tensor[t + 1, :, attr_idx] = predicted_matrix.any(axis=1)

        # raise void bit
        void_entity_mask = ~self._attribute_tensor[t + 1, :, :].any(axis=1)
        self._attribute_tensor[t + 1, void_entity_mask, self.VOID_IDX] = True

    def _predict_next_reward_layer(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        src_slice = self._get_tensor_slice(t, 'attributes')  # (FRAME_STACK_SIZE x N x M)
        transformed_matrix = self._shaper.transform_matrix(src_slice)
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
        attribute_tensor = self._get_env_attribute_tensor()

        self._init_attributes(attribute_tensor)

        # propagate forward
        offset = self.FRAME_STACK_SIZE - 1
        for t in range(offset, offset + self.T):
            self._predict_next_attribute_layer(t)
            self._predict_next_reward_layer(t)

        return self._attribute_tensor

    def get_ball_entity_idx(self, t):
        """
        :param t: time_step you need to look at
        :return: entity_idx of the ball
        """
        entities = self._attribute_tensor[t, :, :]
        row_indices = entities[:, self.BALL_IDX].nonzero()[0]  # returns tuple

        assert row_indices.size <= 1, 'BAD_N_BALLS'
        if row_indices:
            ball_idx = row_indices[0]
        else:
            ball_idx = None
        return ball_idx

    def get_attribute_tensor(self):
        return self._attribute_tensor
