from environment.schema_games.breakout.constants import \
    BRICK_SIZE, ENV_SIZE, DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_PADDLE_SHAPE


class Constants:
    """
    N: number of entities
    M: number of attributes of each entity
    A: number of available actions
    L: number of schemas
    T: size of look-ahead window
    """
    DEBUG = False
    N_BALLS = 1

    VISUALIZE_STATE = False
    VISUALIZE_SCHEMAS = False
    VISUALIZE_INNER_STATE = False
    VISUALIZE_BACKTRACKING = False
    VISUALIZE_REPLAY_BUFFER = False
    LOG_PLANNED_ACTIONS = True

    USE_LEARNED_SCHEMAS = True
    USE_HANDCRAFTED_ATTRIBUTE_SCHEMAS = False
    USE_HANDCRAFTED_REWARD_SCHEMAS = False

    LEARNING_PERIOD = 32
    N_LEARNING_THREADS = 2

    USE_EMERGENCY_REPLANNING = True

    if not DEBUG:
        if ENV_SIZE == 'DEFAULT':
            T = 130  # min 112
            PLANNING_PERIOD = 10
            EMERGENCY_REPLANNING_PERIOD = 30
        elif ENV_SIZE == 'SMALL':
            T = 60  # min 50
            PLANNING_PERIOD = 10
            EMERGENCY_REPLANNING_PERIOD = 8
        else:
            raise AssertionError

        SCREEN_HEIGHT = DEFAULT_HEIGHT
        SCREEN_WIDTH = DEFAULT_WIDTH
        N = SCREEN_WIDTH * SCREEN_HEIGHT
        M = 5
        N_PREDICTABLE_ATTRIBUTES = M - 1
        ACTION_SPACE_DIM = 3
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 2
    else:
        SCREEN_WIDTH = 3
        SCREEN_HEIGHT = 3

        N = 9  # SCREEN_WIDTH * SCREEN_HEIGHT
        M = 2
        T = 16
        ACTION_SPACE_DIM = 3
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 1

    LEARNING_SCHEMA_TOLERANCE = 1e-8
    ADDING_SCHEMA_TOLERANCE = 1e-8

    L = 220
    FILTER_SIZE = 2 * NEIGHBORHOOD_RADIUS + 1
    NEIGHBORS_NUM = FILTER_SIZE ** 2 - 1

    FAKE_ENTITY_IDX = N
    EPSILON = 0

    FRAME_STACK_SIZE = 2
    SCHEMA_VEC_SIZE = FRAME_STACK_SIZE * (M * (NEIGHBORS_NUM + 1)) + ACTION_SPACE_DIM
    TIME_SIZE = FRAME_STACK_SIZE + T

    LEARNING_BATCH_SIZE = FRAME_STACK_SIZE + 1

    # indices of corresponding attributes in entities' vectors
    BALL_IDX = 0
    PADDLE_IDX = 1
    WALL_IDX = 2
    BRICK_IDX = 3
    if not DEBUG:
        VOID_IDX = 4
    else:
        VOID_IDX = 1

    # action indices
    ACTION_NOP = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2

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

    ATTRIBUTE = 'attribute'
    REWARD = 'reward'
    ALLOWED_OBJ_TYPES = {ATTRIBUTE, REWARD}

    DEFAULT_PADDLE_SHAPE = DEFAULT_PADDLE_SHAPE

"""
env changed constants:

BOUNCE_STOCHASTICITY = 0.25
PADDLE_SPEED_DISTRIBUTION[-1] = 0.90
PADDLE_SPEED_DISTRIBUTION[-2] = 0.10
_MAX_SPEED = 2

DEFAULT_BRICK_SHAPE = np.array([8, 4])
DEFAULT_NUM_BRICKS_ROWS = 6
DEFAULT_NUM_BRICKS_COLS = 11
"""
