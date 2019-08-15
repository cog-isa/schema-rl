import numpy as np
from .constants import Constants


class Planner(Constants):
    def __init__(self):
        # (T x 2)
        self.planned_actions = np.empty((self.T, self.ACTION_SPACE_DIM),
                                        dtype=object)

    def _reset_plan(self):
        pass
        #self.planned_actions.clear()

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
                self._backtrace_attribute(precondition, depth+1)
            if not precondition.is_reachable:
                # schema can *never* be reachable, break and try another schema
                schema.is_reachable = False
                break

    def _backtrace_attribute(self, node, depth):
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
                t = self.T - depth
                self.planned_actions[t, :] = schema.action_preconditions
                break
            else:
                self._reset_plan()  # full reset?