import numpy as np
from .constants import Constants
from .graph_utils import Action, Reward


class Planner(Constants):
    def __init__(self, reward_nodes):
        self.planned_actions = np.full(self.T, Action.not_planned_idx, dtype=int)
        self.epsilon = 0.1

        # from SchemaNetwork
        self._reward_nodes = reward_nodes

    def _reset_plan(self):
        self.planned_actions[:] = Action.not_planned_idx

    def _backtrace_schema(self, schema, depth):
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
                self._backtrace_node(precondition, depth + 1)
            if not precondition.is_reachable:
                # schema can *never* be reachable, break and try another schema
                schema.is_reachable = False
                break

    def _backtrace_node(self, node, depth):
        """
        Determines if node is reachable
        is_reachable: can it be certainly activated given the state at t = 0
        """
        # lazy combining schemas by OR -> assuming False
        node.is_reachable = False

        for schema in node.schemas:
            if schema.is_reachable is None:
                self._backtrace_schema(schema, depth)

            if schema.is_reachable:
                # attribute is reachable by this schema
                t = self.T - depth - 1
                self.planned_actions[t] = schema.action_preconditions[0].idx
                break
            else:
                self._reset_plan()  # full reset?

    def _find_closest_reward(self, reward_sign):
        """
        Returns closest reward_node of sign reward_sign
        or None if such reward was not found
        """
        assert (reward_sign in Reward.allowed_signs)

        closest_reward_node = None
        for node in self._reward_nodes[:, Reward.pos_idx]:
            if node.is_feasible:
                closest_reward_node = node
                break

        return closest_reward_node

    def plan_actions(self):
        # find closest positive reward
        pos_reward_node = self._find_closest_reward('pos')
        if pos_reward_node is not None:
            # backtrace from it
            self._backtrace_node(pos_reward_node, 0)
        else:
            # find closest negative reward
            neg_reward_node = self._find_closest_reward('neg')
            if neg_reward_node is not None:
                # backtrace from it
                self._backtrace_node(neg_reward_node, 0)
            else:
                raise AssertionError

        make_random_action = np.random.choice([True, False],
                                              size=1,
                                              p=[self.epsilon, 1 - self.epsilon])

        if make_random_action:
            pass

