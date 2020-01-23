from collections import defaultdict, namedtuple
import numpy as np
from .constants import Constants
from .graph_utils import Attribute, Action, Reward
from .visualizer import NodeMetadata


class Planner(Constants):
    def __init__(self, reward_nodes):
        # from SchemaNetwork
        # (T x REWARD_SPACE_DIM)
        self._reward_nodes = reward_nodes

        # for backtracking state visualizing
        self.curr_target = None
        self.node2triplets = None

        # for backtracking schemas visualizing
        self.schema_vectors = None  # (TIME_SIZE) list of lists of schemas

    def _reset(self):
        self.curr_target = None
        self.node2triplets = None
        self.schema_vectors = []

    def _backtrace_schema(self, schema):
        """
        Determines if schema is reachable
        Keeps track of action path
        is_reachable: can it be certainly activated given the state at t = 0
        """
        # lazy combining preconditions by AND -> assuming True
        schema.is_reachable = True

        for precondition in schema.attribute_preconditions:
            if precondition.is_reachable is None:
                # this node is NOT at t < FRAME_STACK_SIZE (otherwise it would be initialized as reachable)
                # and we have not computed it's reachability yet
                self._backtrace_node(precondition)
            if not precondition.is_reachable:
                # schema can *never* be reachable, break and try another schema
                schema.is_reachable = False
                break

    def _backtrace_node(self, node):
        """
        Determines if node is reachable
        is_reachable: can it be certainly activated given the state at t = 0
        """
        # lazy combining schemas by OR -> assuming False
        node.is_reachable = False

        # avoiding negative rewards
        # neg_schemas for reward at the same time step as node's time step
        # neg_schemas = self._reward_nodes[node.t, Reward.sign2idx['neg']].schemas
        # node.sort_schemas_by_harmfulness(neg_schemas)

        node.sort_schemas_by_priority()

        for schema in node.schemas:
            if schema.is_reachable is None:
                self._backtrace_schema(schema)

            if schema.is_reachable:
                # attribute is reachable by this schema
                node.is_reachable = True
                node.activating_schema = schema
                node.activating_schema.compute_cumulative_actions()

                # for visualizing backtracking
                if self.VISUALIZE_BACKTRACKING:
                    if type(node) is not Reward:
                        self.node2triplets[self.curr_target].append(
                            (node.t, node.entity_idx, node.attribute_idx))
                    else:
                        pass
                        """
                        print('Encountered Reward node during backtracking.')
                        print(vars(node))
                        for s in node.schemas:
                            print(vars(s))
                        """

                # print attribute schemas with filter conditions
                if type(node) is Attribute and \
                        node.attribute_idx in (self.BALL_IDX, self.PADDLE_IDX) and \
                        schema.action_preconditions:

                    metadata = NodeMetadata(node.t, type(node).__name__,
                                            node.attribute_idx if type(node) is Attribute else None)
                    self.schema_vectors.append((schema.vector, metadata))

                break

    def _find_closest_reward(self, reward_sign, search_from):
        """
        Returns closest reward_node of sign reward_sign
        or None if such reward was not found
        """
        assert (reward_sign in Reward.allowed_signs)

        closest_reward_node = None
        reward_idx = Reward.sign2idx[reward_sign]
        for node in self._reward_nodes[search_from:, reward_idx]:
            if node.is_feasible:
                closest_reward_node = node
                break

        return closest_reward_node

    def _find_all_rewards(self, reward_sign):
        """
        Returns all reward nodes of sign reward_sign
        """
        assert (reward_sign in Reward.allowed_signs)
        reward_idx = Reward.sign2idx[reward_sign]

        rewards = []

        for node in self._reward_nodes[:, reward_idx]:
            if node.is_feasible:
                rewards.append(node)

        return rewards

    def _plan_for_rewards(self, reward_sign):
        """
        :param reward_sign: {pos, neg}
        :return: ndarray of len (T)
                 None if cannot plan for this sign of rewards
        """
        target_reward_nodes = []
        print('Trying to plan for {} rewards...'.format(reward_sign))
        planned_actions = None

        rewards = self._find_all_rewards(reward_sign)
        rewards = sorted(rewards, key=lambda node: node.weight, reverse=True)

        for reward_node in rewards:
            target_reward_nodes.append(reward_node)
            print('Found feasible {} reward node with weight {}...'.format(reward_sign, reward_node.weight))

            # backtrace from it
            self.curr_target = reward_node
            self.node2triplets = defaultdict(list)
            self.schema_vectors = []

            self._backtrace_node(reward_node)
            if reward_node.is_reachable:
                print('Actions have been planned successfully!')

                # here planned_actions is len(t-1) List of len(max(x, ACTION_SPACE_DIM)) Lists]
                planned_actions = reward_node.activating_schema.required_cumulative_actions

                # remove actions planned for past
                planned_actions = planned_actions[self.FRAME_STACK_SIZE - 1:]

                # picking first action as a result
                planned_actions = [actions_at_t[0] if actions_at_t else Action.not_planned_idx
                                   for actions_at_t in planned_actions]

                planned_actions = np.array(planned_actions)
                break
            else:
                print('Backtraced {} reward node is unreachable.'.format(reward_sign))
        else:
            print('There are no more feasible {} reward nodes in the graph.'.format(reward_sign))

        return planned_actions, target_reward_nodes

    def plan_actions(self):

        self._reset()
        planned_actions, target_reward_nodes = self._plan_for_rewards('pos')

        if planned_actions is not None:
            pass
            """
            randomness_mask = np.random.choice([True, False],
                                               size=planned_actions.size,
                                               p=[self.EPSILON, 1 - self.EPSILON])
            randomness_size = randomness_mask.sum()

            planned_actions[randomness_mask] = np.random.randint(low=0,
                                                                 high=self.ACTION_SPACE_DIM,
                                                                 size=randomness_size)
            """
        else:
            # can't plan anything
            print('Planner failed to plan.')

        return planned_actions, target_reward_nodes

