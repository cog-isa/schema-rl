class Schema:
    """Grounded schema type."""
    def __init__(self, preconditions):
        '''
        preconditions: list of Nodes
        '''
        self.preconditions = preconditions


class Node:
    def __init__(self):
        """
        #node_type: one of {attribute, action, reward}

        value: bool variable
        schemas: list of schemas
        """
        self.value = None
        self.schemas = None

    def add_schemas(self, schemas):
        self.schemas = schemas

    def get_ancestors(self):
        """
        returns: [L x n_schema_pins] matrix of references to Nodes
        """
        assert(schemas is not None)
        return [a for schema in self.schemas]


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
