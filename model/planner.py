class Planner:
    def __init__(self, T):
        self._T = T
        self.planned_actions = [[] for _ in range(self._T)]

    def _reset_plan(self):
        self.planned_actions.clear()

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
                self._backtrace_attribute(precondition)
            if not precondition.is_reachable:
                # schema can *never* be reachable, break and try another schema
                schema.is_reachable = False
                break

    def _backtrace_attribute(self, node):
        """
        Determines if node is reachable

        is_reachable: can it be certainly activated given the state at t = 0
        """

        # lazy combining schemas by OR -> assuming False
        node.is_reachable = False

        for schema in node.schemas:
            if schema.is_reachable is None:
                self._backtrace_schema(schema)

            if schema.is_reachable:
                # attribute is reachable by this schema
                self.planned_actions.append(schema.action_preconditions)
                break
            else:
                self._reset_plan()  # full reset?