import numpy as np
from model.constants import Constants
from collections import namedtuple


class HardcodedSchemaVectors(Constants):
    BALL_IDX = Constants.BALL_IDX
    PADDLE_IDX = Constants.PADDLE_IDX
    WALL_IDX = Constants.WALL_IDX
    BRICK_IDX = Constants.BRICK_IDX
    VOID_IDX = Constants.VOID_IDX

    Precondition = namedtuple('Precondition', ['time_step', 'di', 'dj', 'entity_type_idx'])
    wall = [
        (Precondition('prev', 0, 0, WALL_IDX),
         Precondition('curr', 0, 0, WALL_IDX)),
    ]
    brick = [
        (Precondition('prev', 0, 0, BRICK_IDX),
         Precondition('curr', 0, 0, BRICK_IDX)),
    ]
    paddle = [
        (Precondition('prev', 0, 0, PADDLE_IDX),
         Precondition('curr', 0, 0, PADDLE_IDX)),
        # paddle growing
        (Precondition('curr', 0, -1, PADDLE_IDX),),  # to right
        (Precondition('curr', 0, 1, PADDLE_IDX),),  # to left
    ]
    ball = [
        # linear movement
        (Precondition('prev', -2, -2, BALL_IDX),
         Precondition('curr', -1, -1, BALL_IDX),
         Precondition('curr', 0, 0, VOID_IDX)),
        (Precondition('prev', -2, 2, BALL_IDX),
         Precondition('curr', -1, 1, BALL_IDX),
         Precondition('curr', 0, 0, VOID_IDX)),
        (Precondition('prev', 2, 2, BALL_IDX),
         Precondition('curr', 1, 1, BALL_IDX),
         Precondition('curr', 0, 0, VOID_IDX)),
        (Precondition('prev', 2, -2, BALL_IDX),
         Precondition('curr', 1, -1, BALL_IDX),
         Precondition('curr', 0, 0, VOID_IDX)),
        (Precondition('prev', -2, 0, BALL_IDX),
         Precondition('curr', -1, 0, BALL_IDX),
         Precondition('curr', 0, 0, VOID_IDX)),
        (Precondition('prev', 2, 0, BALL_IDX),
         Precondition('curr', 1, 0, BALL_IDX),
         Precondition('curr', 0, 0, VOID_IDX)),
        # bounce from paddle
        (Precondition('prev', 0, -2, BALL_IDX),  # left to right
         Precondition('curr', 1, -1, BALL_IDX),
         Precondition('curr', 2, -1, PADDLE_IDX)),
        (Precondition('prev', 0, 2, BALL_IDX),  # right to left
         Precondition('curr', 1, 1, BALL_IDX),
         Precondition('curr', 2, 1, PADDLE_IDX)),
        (Precondition('prev', 0, 0, BALL_IDX),  # upright
         Precondition('curr', 1, 0, BALL_IDX),
         Precondition('curr', 2, 0, PADDLE_IDX)),
        # bounce from wall
        (Precondition('prev', 2, 0, BALL_IDX),  # left wall, bounce bottom-up
         Precondition('curr', 1, -1, BALL_IDX),
         Precondition('prev', 1, -2, WALL_IDX),
         Precondition('curr', 1, -2, WALL_IDX)),
        (Precondition('prev', -2, 0, BALL_IDX),  # left wall, bounce top-down
         Precondition('curr', -1, -1, BALL_IDX),
         Precondition('prev', -1, -2, WALL_IDX),
         Precondition('curr', -1, -2, WALL_IDX)),
        (Precondition('prev', 2, 0, BALL_IDX),  # right wall, bounce bottom-up
         Precondition('curr', 1, 1, BALL_IDX),
         Precondition('prev', 1, 2, WALL_IDX),
         Precondition('curr', 1, 2, WALL_IDX)),
        (Precondition('prev', -2, 0, BALL_IDX),  # right wall, bounce top-down
         Precondition('curr', -1, 1, BALL_IDX),
         Precondition('prev', -1, 2, WALL_IDX),
         Precondition('curr', -1, 2, WALL_IDX)),
    ]

    entity_types = (ball, paddle, wall, brick)

    positive_reward = [
        (Precondition('curr', 0, 0, BALL_IDX),  # attack from left
         Precondition('prev', 1, -1, BALL_IDX),
         Precondition('curr', -1, 0, BRICK_IDX),),
        (Precondition('curr', 0, 0, BALL_IDX),  # attack from right
         Precondition('prev', 1, 1, BALL_IDX),
         Precondition('curr', -1, 0, BRICK_IDX),),
    ]
    negative_reward = [
        (Precondition('curr', 0, 0, BALL_IDX),  # all-in-center (just to fill 1 schema)
         Precondition('curr', 0, 0, PADDLE_IDX),
         Precondition('curr', 0, 0, WALL_IDX),
         Precondition('curr', 0, 0, BRICK_IDX),)
    ]
    rewards = [positive_reward, negative_reward]

    @classmethod
    def convert_filter_offset_to_schema_vec_idx(cls, time_step, di, dj, entity_type_idx):
        assert time_step in ('prev', 'curr')

        i = cls.NEIGHBORHOOD_RADIUS + di
        j = cls.NEIGHBORHOOD_RADIUS + dj
        filter_idx = i * cls.FILTER_SIZE + j

        mid = (cls.NEIGHBORS_NUM + 1) // 2
        if filter_idx == mid:
            vec_entity_idx = 0
        elif filter_idx < mid:
            vec_entity_idx = filter_idx + 1
        else:
            vec_entity_idx = filter_idx

        if time_step == 'curr':
            vec_entity_idx += cls.NEIGHBORS_NUM + 1

        vec_idx = vec_entity_idx * cls.M + entity_type_idx
        return vec_idx

    @classmethod
    def make_schema_vec(cls, preconditions):
        vec = np.full(cls.SCHEMA_VEC_SIZE, False, dtype=bool)
        for precondition in preconditions:
            idx = cls.convert_filter_offset_to_schema_vec_idx(precondition.time_step,
                                                              precondition.di,
                                                              precondition.dj,
                                                              precondition.entity_type_idx)
            vec[idx] = True
        return vec

    @classmethod
    def make_target_schema_matrices(cls, prediction_targets):
        A = []
        for target in prediction_targets:
            A_i = []
            for schema_preconditions in target:
                vec = cls.make_schema_vec(schema_preconditions)
                A_i.append(vec)
            A.append(A_i)
        A = [np.stack(A_i, axis=0).T for A_i in A]
        return A

    @classmethod
    def gen_schema_matrices(cls):
        W = cls.make_target_schema_matrices(cls.entity_types)
        R = cls.make_target_schema_matrices(cls.rewards)
        return W, R


class TestFSS1:
    W1 = np.reshape(np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]), (11, 1)).astype(bool)
    W = [W1]

    R1 = np.reshape(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), (11, 1)).astype(bool)
    R2 = np.reshape(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (11, 1)).astype(bool)

    R = [R1, R2]


class TestFSS2:
    W1 = np.reshape(np.array([0, 1, 0, 0, 1, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 1, 0]), (21, 1)).astype(bool)
    W = [W1]

    R1 = np.reshape(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 1]), (21, 1)).astype(bool)

    R2 = np.reshape(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0]), (21, 1)).astype(bool)

    R = [R1, R2]