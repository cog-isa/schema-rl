import numpy as np
from .constants import Constants

from environment.schema_games.breakout.constants import \
    CLASSIC_BACKGROUND_COLOR, CLASSIC_BALL_COLOR, CLASSIC_BRICK_COLORS, \
    CLASSIC_PADDLE_COLOR, CLASSIC_WALL_COLOR


class Visualizer(Constants):
    def __init__(self):
        self.COLOR_DEPTH = 3

        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
        self._attribute_tensor = None

        self._color_map = {
            0: CLASSIC_BALL_COLOR,
            1: CLASSIC_PADDLE_COLOR,
            2: CLASSIC_WALL_COLOR,
            3: CLASSIC_BRICK_COLORS[0],
            self.M: CLASSIC_BACKGROUND_COLOR
        }

    def set_attribute_tensor(self, attribute_tensor):
        self._attribute_tensor = attribute_tensor

    def _gen_image(self, t):
        row_indices, col_indices = np.where(self._attribute_tensor[t])
        assert row_indices.size == np.unique(row_indices).size, \
            'CONFLICT: several bits per pixel'

        # print('row_indices', row_indices.shape)
        # print('col_indices', col_indices.shape)

        flat_pixels = np.full((self.N, self.COLOR_DEPTH), self.M, dtype=np.uint8)

        colors = np.array([self._color_map[col_idx] for col_idx in col_indices])

        # print('colors', colors.shape)
        # print('target', flat_pixels[row_indices, :].shape)
        # print()
        if row_indices.size > 0:
            flat_pixels[row_indices, :] = colors

        pixels = flat_pixels.reshape((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.COLOR_DEPTH))

    def gen_images(self):
        for t in range(self._attribute_tensor.shape[0]):
            image = self._gen_image(t)

    def check_correctness(self):
        print('Checking correctness of given and predicted states...')

        print('Given states:')
        for t in range(self._attribute_tensor.shape[0]):
            if t == self.FRAME_STACK_SIZE:
                print('Predicted states:')
            n_conflicts = (self._attribute_tensor[t].sum(axis=1) > 1).sum()
            print('t: {}, n_conflicts: {}'.format(t, n_conflicts))
