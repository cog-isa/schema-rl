import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
import time
from .graph_utils import Attribute, Action


class FeatureMatrix:
    def __init__(self, env, shape=(117, 94), attrs_num=53, window_size=2, action_space=3):
        self.shape = shape
        self.matrix = np.zeros((shape[0] * shape[1], attrs_num))
        self.attrs_num = attrs_num
        self.window_size = window_size
        self.action_space = action_space
        self.entities_num = shape[0] * shape[1]

        self.ball_attr = 0
        self.paddle_attr = 1
        self.wall_attr = 2
        self.brick_attr = 3

        self.planned_action = None

        for ball in env.balls:
            if ball.is_entity:
                for state, eid in env.parse_object_into_pixels(ball):
                    pos = list(state.keys())[0][1]
                    ind = self.transform_pos_to_index(pos)
                    self.matrix[ind][self.ball_attr] = 1

        if env.paddle.is_entity:
            for state, eid in env.parse_object_into_pixels(env.paddle):
                # TODO: add parts of paddle
                pos = list(state.keys())[0][1]
                for i in range(-11, 12):
                    for j in range(-1, 3):
                        ind = self.transform_pos_to_index((pos[0] + j, pos[1] + i))
                        self.matrix[ind][self.paddle_attr] = 1

                print(eid)

        for wall in env.walls:
            if wall.is_entity:
                for state, eid in env.parse_object_into_pixels(wall):
                    pos = list(state.keys())[0][1]
                    ind = self.transform_pos_to_index(pos)
                    self.matrix[ind][self.wall_attr] = 1

        for brick in env.bricks:
            # TODO: add parts of bricks
            if brick.is_entity:
                for state, eid in env.parse_object_into_pixels(brick):
                    pos = list(state.keys())[0][1]
                    ind = self.transform_pos_to_index(pos)
                    self.matrix[ind][self.brick_attr] = 1

    def transform_pos_to_index(self, pos):
        return pos[0] * self.shape[1] + pos[1]

    def transform_index_to_pos(self, index):
        return index // self.shape[1], index % self.shape[1]

    def get_neighbours(self, ind, action, matrix=None, add_all_actions=False):

        if matrix is None:
            matrix = self.matrix

        pos = self.transform_index_to_pos(ind)
        x = pos[0]
        y = pos[1]
        res = []

        if add_all_actions:
            action_vec = np.ones(self.action_space)
        else:
            action_vec = np.eye(self.action_space)[action - 1]

        zeros = np.zeros(self.attrs_num)
        zero_attributes = [None for _ in range(self.attrs_num)]

        entity_indices = []

        for i in range(-self.window_size, self.window_size):
            for j in range(-self.window_size, self.window_size):
                if x + i < 0 or x + i >= self.shape[0] or y + j < 0 or y + j >= self.shape[1]:
                    res.append(zeros)
                    entity_indices.append(zero_attributes)  # adding empty entity
                else:
                    idx = self.transform_pos_to_index([x + i, y + j])
                    res.append(matrix[idx])
                    entity_indices.append(
                        [Attribute(idx, attr_idx) for attr_idx in range(self.attrs_num)]
                    )  # adding entity's index

        res.append(action_vec)
        actions = [Action(idx) for idx in range(self.action_space)]
        entity_indices.append(actions)

        return np.concatenate(res), entity_indices

    def get_attribute_matrix(self):
        return self.matrix.copy()

    def transform_matrix(self, custom_matrix=None, add_all_actions=False):
        if custom_matrix is not None:
            matrix = custom_matrix
        else:
            matrix = self.matrix

        transformed_matrix = []
        idx_matrix = []
        for i in range(0, self.entities_num):
            transformed_vec, entity_indices = \
                self.get_neighbours(i, self.planned_action, matrix=matrix, add_all_actions=add_all_actions)
            transformed_matrix.append(transformed_vec)
            idx_matrix.append(entity_indices)

        return np.array([transformed_matrix]), np.array(idx_matrix)


if __name__ == '__main__':
    env = StandardBreakout()
    state = env.reset()
    start = time.time()
    mat = FeatureMatrix(env, attrs_num=4)
    end = time.time()
    print("--- %s seconds ---" % (end - start))
    start = time.time()
    # TODO: make it faster (bin type of data, but relaxed LP optimisation???)
    mat.planned_action = 1
    X = mat.transform_matrix()[0]
    end = time.time()
    print("--- %s seconds ---" % (end - start))
    X = np.array(X)
    print(X.shape)
    print(X)
