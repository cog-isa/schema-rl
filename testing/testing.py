import numpy as np
from model.constants import Constants


class HardcodedSchemaVectors(Constants):
    wall = [
        ((0, 0), (0, 0)),
    ]

    brick = [
        ((0, 0), (0, 0)),
    ]

    paddle = [
        ((0, 0), (0, 0)),
    ]

    ball = [
        ((-2, -2), (-1, -1)),
    ]

    object_types = (ball, paddle, wall, brick)

    @classmethod
    def convert_filter_offset_to_schema_vec_idx(cls, di, dj, time_step, entity_type_idx):
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

        vec_idx = vec_entity_idx * cls.M + entity_type_idx

        if time_step == 'curr':
            vec_idx += cls.NEIGHBORS_NUM + 1

        return vec_idx

    @classmethod
    def gen_attribute_schema_matrices(cls):
        W = []
        for entity_type_idx, object_type in enumerate(cls.object_types):
            W_i = []
            for record in object_type:
                vec = np.full(cls.SCHEMA_VEC_SIZE, False, dtype=bool)
                for time_record, time_param in zip(record, ('prev', 'curr')):
                    di = time_record[0]
                    dj = time_record[1]
                    idx = cls.convert_filter_offset_to_schema_vec_idx(di, dj, time_param, entity_type_idx)
                    vec[idx] = True
                W_i.append(vec)
            W.append(W_i)

        W = [np.stack(W_i, axis=0).T for W_i in W]
        return W


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