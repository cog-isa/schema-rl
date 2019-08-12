class Schema:
    """Grounded schema type."""
    def __init__(self, attribute_preconditions, action_preconditions):
        """
        preconditions: list of Nodes
        """
        self.attribute_preconditions = attribute_preconditions
        self.action_preconditions = action_preconditions
        self.preconditions = self.attribute_preconditions + self.action_preconditions


class Node:
    def __init__(self):
        """
        value: bool variable
        schemas: list of schemas
        is_discovered: has node been seen during graph traversal
        """
        self.value = None
        self.schemas = None
        self.is_discovered = False

    def add_schemas(self, schemas):
        self.schemas = schemas

    def get_ancestors(self):
        """
        returns: [L x n_schema_pins] matrix of references to Nodes
        """
        assert(schemas is not None)
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


class Action(Node):
    def __init__(self, action_idx):
        """
        action_idx: action unique idx
        """
        self.action_idx = action_idx
        super().__init__()


class Reward(Node):
    def __init__(self, sign):
        assert(sign in ('pos', 'neg'))
        self.sign = sign
        super().__init__()
