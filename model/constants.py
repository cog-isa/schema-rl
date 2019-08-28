class Constants:
    """
    N: number of entities
    M: number of attributes of each entity
    A: number of available actions
    L: number of schemas
    T: size of look-ahead window
    """
    DEBUG = False

    if DEBUG:
        SCREEN_WIDTH = 3
        SCREEN_HEIGHT = 3

        N = 9  # SCREEN_WIDTH * SCREEN_HEIGHT
        M = 1
        T = 3
        ACTION_SPACE_DIM = 2
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 1
    else:
        SCREEN_WIDTH = 94
        SCREEN_HEIGHT = 117

        N = SCREEN_WIDTH * SCREEN_HEIGHT
        M = 4
        T = 3
        ACTION_SPACE_DIM = 2
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 2
        NEIGHBORS_NUM = 24
