import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
import time
from .graph_utils import MetaFactory
from .constants import Constants


class FeatureMatrix(Constants):
    def __init__(self, env, shape=(117, 94), attrs_num=53, window_size=2, action_space=3):
        self.shape = shape
        self.matrix = np.zeros((shape[0]*shape[1], attrs_num))
        self.entities_num = shape[0] * shape[1]
        self.attrs_num = attrs_num
        self.window_size = window_size
        self.action_space = action_space

        self.ball_attr = 0
        self.paddle_attr = 1
        self.wall_attr = 2
        self.brick_attr = 3

        self.planned_action = None
        self._meta_factory = MetaFactory()

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
                        ind = self.transform_pos_to_index((pos[0]+j, pos[1]+i))
                        self.matrix[ind][self.paddle_attr] = 1


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
        return pos[0]*self.shape[1] + pos[1]

    def transform_index_to_pos(self, index):
        return index//self.shape[1], index % self.shape[1]

    def transform_pos_to_idx(self, pos):
        """
        :param pos: tuple of (row_idx, col_idx)
        :return: idx of pixel that runs by rows left2right
        """
        i = pos[0]
        j = pos[1]
        return i * self.SCREEN_WIDTH + j

    def transform_idx_to_pos(self, idx):
        """
        :param idx: idx of pixel that runs by rows left2right
        :return: tuple of (row_idx, col_idx)
        """
        i = idx // self.SCREEN_WIDTH
        j = idx % self.SCREEN_WIDTH
        return (i, j)

    def get_attribute_matrix(self):
        return self.matrix.copy()

    def get_neighbours(self, ind, action, matrix=None):

        if matrix is None:
            matrix = self.matrix

        x, y = self.transform_idx_to_pos(ind)

        res = []
        metadata_row = []

        action_vec = np.full(self.ACTION_SPACE_DIM, True)
        zeros = np.full(self.M, False)

        for i in range(-self.window_size, self.window_size):
            for j in range(-self.window_size, self.window_size):
                if x + i < 0 or x + i >= self.SCREEN_WIDTH or y+j < 0 or y+j >= self.SCREEN_HEIGHT:
                    res.append(zeros)
                    meta_fake_entity = self._meta_factory.gen_meta_entity(0, fake=True)
                    metadata_row.extend(meta_fake_entity)
                else:
                    entity_idx = self.transform_pos_to_idx([x + i, y + j])
                    res.append(matrix[entity_idx].astype(bool))

                    meta_entity = self._meta_factory.gen_meta_entity(entity_idx, fake=False)
                    metadata_row.extend(meta_entity)

        res.append(action_vec)

        meta_actions = self._meta_factory.gen_meta_actions()
        metadata_row.extend(meta_actions)

        return np.concatenate(res), metadata_row

    def transform_matrix(self, custom_matrix=None, add_all_actions=False, output_format='attribute'):

        if custom_matrix is not None:
            matrix = custom_matrix
        else:
            matrix = self.matrix

        transformed_matrix = []
        metadata_matrix = []

        if output_format == 'attribute':
            for i in range(0, self.N):
                transformed_vec, metadata_row = \
                    self.get_neighbours(i, self.planned_action, matrix=matrix)
                transformed_matrix.append(transformed_vec)
                metadata_matrix.append(metadata_row)
        elif output_format == 'reward':
            # should return (1 x (NM + A)) matrix
            print('reaching stub!')
            shape = (self.N * self.M + self.ACTION_SPACE_DIM)
            transformed_matrix = np.full((1, shape), True)
            metadata_matrix = []
            for entity_idx in range(self.N):
                metadata_matrix.extend(
                    self._meta_factory.gen_meta_entity(entity_idx, fake=False)
                )
            metadata_matrix.extend(
                self._meta_factory.gen_meta_actions()
            )
            metadata_matrix = [metadata_matrix]
            # raise NotImplementedError

        transformed_matrix = np.array(transformed_matrix)
        metadata_matrix = np.array(metadata_matrix)

        return transformed_matrix, metadata_matrix

    def get_neighbours_with_action(self, ind, action, matrix=None, add_all_actions=False):

        if matrix is None:
            matrix = self.matrix

        pos = self.transform_index_to_pos(ind)
        x = pos[0]
        y = pos[1]
        res = []

        if add_all_actions:
            action_vec = np.ones(self.action_space)
        else:
            action_vec = np.eye(self.action_space)[action-1]

        zeros = np.zeros(self.attrs_num)

        for i in range(-self.window_size, self.window_size+1):
            for j in range(-self.window_size, self.window_size+1):
                if x + i < 0 or x + i >= self.shape[0] or y+j < 0 or y+j >= self.shape[1]:
                    res.append(zeros)
                else:
                    res.append(matrix[self.transform_pos_to_index([x + i, y + j])])

        res.append(action_vec)

        return np.concatenate(res)

    def transform_matrix_with_action(self, action, custom_matrix=None, add_all_actions=False):
        if custom_matrix is not None:
            matrix = custom_matrix
        else:
            matrix = self.matrix

        return np.array([self.get_neighbours_with_action(i, action, matrix=matrix, add_all_actions=add_all_actions)
                         for i in range(0, self.shape[0]*self.shape[1])])


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
