class Schema:
    """Grounded schema type."""

    def __init__(self, attribute_preconditions, action_preconditions):
        """
        preconditions: list of Nodes
        """
        self.is_reachable = None
        self.attribute_preconditions = attribute_preconditions
        self.action_preconditions = action_preconditions
        self.ancestor_actions = None


class Node:
    def __init__(self):
        """
        value: bool variable
        schemas: list of schemas
        is_discovered: has node been seen during graph traversal
        """
        self.is_feasible = None
        self.is_reachable = None
        # reachable by this required actions
        self.required_actions = [None for _ in range()]
        self.value = None
        self.schemas = None
        self.ancestor_actions = None

    def add_schemas(self, schemas):
        self.schemas = schemas

    def get_ancestors(self):
        """
        returns: [L x n_schema_pins] matrix of references to Nodes
        """
        assert (schemas is not None)
        return [schema.preconditions for schema in self.schemas]


class Attribute(Node):
    def __init__(self, entity_idx, attribute_idx):
        """
        entity_idx: entity unique idx
        attribute_idx: attribute index in entity's attribute vector
        """
        self.entity_idx = entity_idx
        self.attribute_idx = attribute_idx
        super().__init__()


class Action:
    def __init__(self, name, time_step):
        """
        action_idx: action unique idx
        """
        self.name = name
        self.time_step = time_step


class Reward(Node):
    def __init__(self, sign):
        assert (sign in ('pos', 'neg'))
        self.sign = sign
        super().__init__()
