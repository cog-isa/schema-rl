import numpy as np
from .constants import Constants
from .graph_utils import Schema, Node, Attribute, Action, Reward
from .tensor_handler import TensorHandler
from .planner import Planner
from .visualizer import Visualizer


class SchemaNetwork(Constants):
    def __init__(self):
        """
        :param W: list of M matrices, each of [(MR + A) x L] shape
        :param R: list of 2 matrices, each of [(MN + A) x L] shape, 1st - pos, 2nd - neg
        """
        self._W = None
        self._R = None
        self._R_weights = None

        self._attribute_nodes = None  # tensor ((FRAME_STACK_SIZE + T) x N x M)
        self._action_nodes = None  # tensor ((FRAME_STACK_SIZE + T) x ACTION_SPACE_DIM)
        self._reward_nodes = None  # tensor ((FRAME_STACK_SIZE + T) x REWARD_SPACE_DIM)

        self._gen_attribute_nodes()
        self._gen_action_nodes()
        self._gen_reward_nodes()

        self._tensor_handler = TensorHandler(self._attribute_nodes,
                                             self._action_nodes, self._reward_nodes)
        self._planner = Planner(self._reward_nodes)
        self._visualizer = Visualizer(self._tensor_handler, self._planner, self._attribute_nodes)
        self._iter = None

    def set_weights(self, W, R, R_weights=None):
        self._process_input(W, R)
        self._print_input_stats(W)

        if R_weights is None:
            R_weights = np.ones(R[0].shape[1], dtype=np.float)

        self._W = W
        self._R = R
        self._R_weights = R_weights
        self._tensor_handler.set_weights(W, R, R_weights)

    def _process_input(self, W, R):
        assert len(W) == self.M - 1, 'BAD_W_NUM'
        # drop first schema - it's bad
        for i in range(len(W)):
            if W[i][:, 0].all():
                W[i] = W[i][:, 1:].copy()

        required_matrix_shape = (self.SCHEMA_VEC_SIZE, self.L)
        for matrix in (W + R):
            assert matrix.dtype == bool, 'BAD_MATRIX_DTYPE'
            assert matrix.ndim == 2, 'BAD_MATRIX_NDIM'
            assert (matrix.shape[0] == required_matrix_shape[0]
                    and matrix.shape[1] <= required_matrix_shape[1]), 'BAD_MATRIX_SHAPE'
            assert matrix.size, 'EMPTY_MATRIX'

    def _print_input_stats(self, W):
        print('Numbers of schemas in W are: ', end='')
        for idx, w in enumerate(W):
            print('{}'.format(w.shape[1]), end='')
            if idx != len(W) - 1:
                print(' / ', end='')
        print()

    def set_curr_iter(self, iter):
        self._iter = iter

    def _gen_attribute_node_matrix(self, t):
        n_rows = self.N
        n_cols = self.M
        matrix = [
            [Attribute(entity_idx, attribute_idx, t) for attribute_idx in range(n_cols)]
            for entity_idx in range(n_rows)
        ]
        return matrix

    def _gen_attribute_nodes(self):
        tensor = [self._gen_attribute_node_matrix(t) for t in range(self.FRAME_STACK_SIZE + self.T)]
        self._attribute_nodes = np.array(tensor)

    def _init_attribute_nodes(self):
        for node in self._attribute_nodes.flat:
            node.reset()

    def _gen_action_nodes(self):
        action_nodes = [
            [Action(idx, t=t) for idx in range(self.ACTION_SPACE_DIM)]
            for t in range(self.FRAME_STACK_SIZE + self.T)
        ]
        self._action_nodes = np.array(action_nodes)

    def _gen_reward_nodes(self):
        reward_nodes = [
            [Reward(idx, t=t) for idx in range(self.REWARD_SPACE_DIM)]
            for t in range(self.FRAME_STACK_SIZE + self.T)
        ]
        self._reward_nodes = np.array(reward_nodes)

    def _init_reward_nodes(self):
        for node in self._reward_nodes.flat:
            node.reset()

    def plan_actions(self, frame_stack):
        if len(frame_stack) < self.FRAME_STACK_SIZE:
            print('Small ENTITIES_STACK. Abort.')
            return None

        self._init_attribute_nodes()
        self._init_reward_nodes()

        # instantiate schemas, determine nodes feasibility
        self._tensor_handler.forward_pass(frame_stack)

        # visualizing
        self._visualizer.set_iter(self._iter)
        if self.VISUALIZE_SCHEMAS:
            self._visualizer.visualize_schemas(self._W, self._R)
        if self.VISUALIZE_INNER_STATE:
            self._visualizer.visualize_predicted_entities(check_correctness=False)

        # planning actions
        actions, target_reward_nodes = self._planner.plan_actions()

        #for t in range(self.TIME_SIZE):
        #    self._tensor_handler.check_entities_for_correctness(t)

        if self.VISUALIZE_BACKTRACKING_SCHEMAS:
            self._visualizer.visualize_backtracking_schemas(self._planner.schema_vectors)

        if self.VISUALIZE_BACKTRACKING_INNER_STATE:
            if target_reward_nodes:
                self._visualizer.visualize_backtracking(target_reward_nodes,
                                                        self._planner.node2triplets)
                #for t in range(self.TIME_SIZE):
                    #self._tensor_handler.check_entities_for_correctness(t)

                #self._visualizer.log_balls_at_backtracking(target_reward_nodes[0])

        return actions
