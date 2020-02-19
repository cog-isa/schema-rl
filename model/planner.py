from collections import defaultdict, namedtuple
import functools
import itertools
import numpy as np
from .constants import Constants
from .graph_utils import Attribute, Action, Reward, Constraint
from .visualizer import NodeMetadata


class Planner(Constants):
    def __init__(self, reward_nodes):
        # (T x REWARD_SPACE_DIM) from SchemaNetwork
        self._reward_nodes = reward_nodes

        self._joint_constraints = [Constraint() for _ in range(self.TIME_SIZE)]

        # for backtracking state visualizing
        self.curr_target = None
        self.node2triplets = None
        # for backtracking schemas visualizing
        self.schema_vectors = None  # (TIME_SIZE) list of lists of schemas

    def _reset(self):
        for c in self._joint_constraints:
            c.reset()

        self.curr_target = None
        self.node2triplets = None
        self.schema_vectors = []

    def _backtrace_schema(self, schema):
        """
        Determines if schema is reachable
        is_reachable: can it be certainly activated under current joint_constraints,
            provided that schema.action_preconditions already satisfy those constraints
        """
        # lazy combining preconditions by AND -> assuming True
        schema.is_reachable = True

        for precondition in schema.attribute_preconditions:
            if precondition.is_reachable is None:
                # this node is NOT at t < FRAME_STACK_SIZE (otherwise it would be initialized as reachable)
                # and we have not computed it's reachability yet
                self._backtrace_node(precondition)
            if not precondition.is_reachable:
                # schema cannot be reachable under current joint constraints,
                # break and try another schema
                schema.is_reachable = False
                break

    def _backtrace_node_by_schema(self, node, schema):
        # assert schema.is_reachable is None, 'SCHEMA_MULTIPLE_BACKTRACKING'
        self._backtrace_schema(schema)

        if schema.is_reachable:
            # attribute is reachable by this schema
            node.is_reachable = True
            node.activating_schema = schema

            # conflicts of actions can occur here *only* during replanning calls
            # assuming they are satisfied, actual mutation of joint constraints is
            # performed in replanning function

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

            """ old vis
            # for visualizing backtracking inner state
            if self.VISUALIZE_BACKTRACKING_INNER_STATE and type(node) is not Reward:
                self.node2triplets[self.curr_target].append(
                    (node.t, node.entity_idx, node.attribute_idx))

            # for visualizing backtracking schemas
            if self.VISUALIZE_BACKTRACKING_SCHEMAS and \
                    sum(a.idx != 0 for a in schema.action_preconditions):
                metadata = NodeMetadata(node.t, str(type(node).__name__),
                                        node.attribute_idx if type(node) is Attribute else None)
                self.schema_vectors.append((schema.vector, metadata))
            """
    def _backtrace_node_by_set_of_schemas(self, node, schemas):
        for schema in schemas:
            self._backtrace_node_by_schema(node, schema)
            if node.is_reachable:
                break

    def _backtrace_node(self, node, desired_constraint=None):
        """
        Determines if node is reachable
        is_reachable: can it be certainly activated under current joint_constraints
        desired_constraint: required action at (node.t - 1) for *this* node to be activated
            do not replan when present
        """

        # when replanning this node, constraint at time t is changing,
        # thus reset of this not-trusted-node's status is needed
        # otherwise lazy combining schemas by OR -> assuming False
        node.is_reachable = False

        # actual replanning of this node to desired_constraint
        if desired_constraint is not None:
            target_schemas = node.schemas[desired_constraint]
            self._backtrace_node_by_set_of_schemas(node, target_schemas)
            return

        # try to activate node using schema without action precondition
        target_schemas = node.schemas[None]
        self._backtrace_node_by_set_of_schemas(node, target_schemas)
        if node.is_reachable:
            return

        # check for constraint
        constraint = self._joint_constraints[node.t - 1]

        # pick any schema if there is no constraint
        if constraint.action_idx is None:
            target_schemas = itertools.chain.from_iterable(
                [v for k, v in node.schemas.items() if k is not None])
            self._backtrace_node_by_set_of_schemas(node, target_schemas)

            # set new constraint
            if node.is_reachable:
                schema_action_idx = node.activating_schema.action_preconditions[0].idx
                constraint.action_idx = schema_action_idx
                constraint.committed_nodes.add(node)

            return

        # try to activate node satisfying joint constraint at time (node.t - 1)
        target_schemas = node.schemas[constraint.action_idx]
        self._backtrace_node_by_set_of_schemas(node, target_schemas)

        if node.is_reachable:
            # add committed node to current constraint and exit
            constraint.committed_nodes.add(node)
            return

        # with each loop we stray further from God
        # there is constraint on this level and no schema can be activated without violating it
        # replan all nodes that are committed to current constraint
        print('Cannot activate schema without replanning.')

        # find actions, acceptable by conflicting nodes
        negotiated_actions = functools.reduce(
            set.intersection,
            (n.acceptable_constraints for n in constraint.committed_nodes),
            node.acceptable_constraints)

        for action in negotiated_actions:
            is_success = self._replan_nodes_with_constraint(
                {node} | constraint.committed_nodes, action, node.t)
            if is_success:
                print('Replanning was successfull.')
                break
            else:
                print('Replanning failed.')

    def _replan_nodes_with_constraint(self, nodes, action, layer_t):
        print('Replanning committed nodes to action: {}'.format(action))
        is_success = True

        for node in nodes:
            self._backtrace_node(node, desired_constraint=action)
            if not node.is_reachable:
                node.is_reachable = True
                is_success = False
                break

        if is_success:
            # perform actual mutation of joint constraints
            # when all replanning has been successfully executed
            curr_constraint = self._joint_constraints[layer_t - 1]
            curr_constraint.action = action
            curr_constraint.committed_nodes.clear()
            curr_constraint.committed_nodes.update(nodes)

        return is_success

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
                # planned_actions = reward_node.activating_schema.required_cumulative_actions

                constraints_before_reward = self._joint_constraints[:reward_node.t]
                planned_actions = [c.action_idx for c in constraints_before_reward]
                planned_actions = [a if a is not None else 0 for a in planned_actions]

                # remove actions planned for past
                planned_actions = planned_actions[self.FRAME_STACK_SIZE - 1:]

                # picking first action as a result
                # planned_actions = [actions_at_t[0] if actions_at_t else Action.not_planned_idx
                #                    for actions_at_t in planned_actions]

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

