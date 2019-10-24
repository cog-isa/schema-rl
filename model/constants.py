class Constants:
    """
    N: number of entities
    M: number of attributes of each entity
    A: number of available actions
    L: number of schemas
    T: size of look-ahead window
    """
    DEBUG = True

    VISUALIZE_SCHEMAS = True
    VISUALIZE_INNER_STATE = True

    if not DEBUG:
        SCREEN_WIDTH = 94
        SCREEN_HEIGHT = 117

        N = SCREEN_WIDTH * SCREEN_HEIGHT
        M = 4
        T = 3
        ACTION_SPACE_DIM = 2
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 2
    else:
        SCREEN_WIDTH = 3
        SCREEN_HEIGHT = 3

        N = 9  # SCREEN_WIDTH * SCREEN_HEIGHT
        M = 1
        T = 50
        ACTION_SPACE_DIM = 3
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 1

    L = 1000
    FILTER_SIZE = 2 * NEIGHBORHOOD_RADIUS + 1
    NEIGHBORS_NUM = FILTER_SIZE ** 2 - 1

    FAKE_ENTITY_IDX = N
    EPSILON = 0
    FRAME_STACK_SIZE = 2

    SCHEMA_VEC_SIZE = FRAME_STACK_SIZE * (M * (NEIGHBORS_NUM + 1)) + ACTION_SPACE_DIM

    # indices of corresponding attributes in entities' vectors
    BALL_IDX = 0
    PADDLE_IDX = 1
    WALL_IDX = 2
    BRICK_IDX = 3

    ENTITY_NAMES = {
        BALL_IDX: 'BALL',
        PADDLE_IDX: 'PADDLE',
        WALL_IDX: 'WALL',
        BRICK_IDX: 'BRICK',
    }

    REWARD_NAMES = {
        0: 'POSITIVE',
        1: 'NEGATIVE',
    }
