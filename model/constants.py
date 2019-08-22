class Constants:
    """
    N: number of entities
    M: number of attributes of each entity
    A: number of available actions
    L: number of schemas
    T: size of look-ahead window
    """
    SCREEN_WIDTH = 10  # 94
    SCREEN_HEIGHT = 10  # 117

    N = SCREEN_WIDTH * SCREEN_HEIGHT
    M = 10  # 53
    A = 3
    L = 5
    T = 3  # 50
    ACTION_SPACE_DIM = 2
    REWARD_SPACE_DIM = 2
