import numpy as np
from .constants import Constants
from .graph_utils import Action, Reward


class Planner(Constants):
    def __init__(self, reward_nodes):
        # from SchemaNetwork
        # (T x REWARD_SPACE_DIM)
        self._reward_nodes = reward_nodes

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
                # this node is NOT at t = 0 AND we have not computed it's value
                # dfs over precondition's schemas
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

        for schema in node.schemas:
            if schema.is_reachable is None:
                self._backtrace_schema(schema)

            if schema.is_reachable:
                # attribute is reachable by this schema
                node.is_reachable = True
                node.activating_schema = schema
                node.activating_schema.compute_cumulative_actions()
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

    def _plan_for_rewards(self, reward_sign):
        """
        :param reward_sign: {pos, neg}
        :return: ndarray of len (T)
                 None if cannot plan for this sign of rewards
        """
        print('trying to plan for {} rewards...'.format(reward_sign))
        planned_actions = None

        search_from = 0
        while search_from < self.T:
            reward_node = self._find_closest_reward(reward_sign, search_from)

            if reward_node is None:
                print('cannot find more {} reward nodes'.format(reward_sign))
                break

            print('found feasible {} reward node, starting to backtrace it...'.format(reward_sign))

            search_from = reward_node.t + 1

            # backtrace from it
            self._backtrace_node(reward_node)
            if reward_node.is_reachable:
                print('actions for reaching target {} reward node have been found successfully!'.format(reward_sign))
                # actions for reaching target reward are planned
                planned_actions = reward_node.activating_schema.required_cumulative_actions

                # here planned_actions is len(t-1) List of len(max(x, ACTION_SPACE_DIM)) Lists]
                # picking FIRST action as a result
                planned_actions = [actions_at_t[0] if actions_at_t else Action.not_planned_idx
                                   for actions_at_t in planned_actions]
                planned_actions = np.array(planned_actions)
                break

            print('backtraced {} reward node is unreachable'.format(reward_sign))

        return planned_actions

    def plan_actions(self):

        planned_actions = self._plan_for_rewards('pos')
        if planned_actions is None:
            # no positive rewards are reachable from current state,
            # trying to find closest negative reward
            # do NOT backtrace negative reward
            #planned_actions = self._plan_for_rewards('neg')
            pass

        if planned_actions is not None:
            randomness_mask = np.random.choice([True, False],
                                               size=planned_actions.size,
                                               p=[self.EPSILON, 1 - self.EPSILON])
            randomness_size = randomness_mask.sum()

            planned_actions[randomness_mask] = np.random.randint(low=0,
                                                                 high=self.ACTION_SPACE_DIM,
                                                                 size=randomness_size)
        else:
            # can't plan anything
            print('Planner failed to plan, returning random actions.')
            planned_actions = np.random.randint(low=0,
                                                high=self.ACTION_SPACE_DIM,
                                                size=self.T)

        return planned_actions

