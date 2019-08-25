class Constants:
    """
    N: number of entities
    M: number of attributes of each entity
    A: number of available actions
    L: number of schemas
    T: size of look-ahead window
    """
    DEBUG = False

    SCREEN_WIDTH = 3  # 94
    SCREEN_HEIGHT = 3  # 117

    N = 9  # SCREEN_WIDTH * SCREEN_HEIGHT
    M = 1  # 53
    L = 1
    T = 3  # 50
    ACTION_SPACE_DIM = 2
    REWARD_SPACE_DIM = 2

    NEIGHBORHOOD_RADIUS = 1
