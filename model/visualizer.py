import numpy as np
from PIL import Image

from .constants import Constants

from environment.schema_games.breakout.constants import \
    CLASSIC_BACKGROUND_COLOR, CLASSIC_BALL_COLOR, CLASSIC_BRICK_COLORS, \
    CLASSIC_PADDLE_COLOR, CLASSIC_WALL_COLOR


class Visualizer(Constants):
    def __init__(self):
        self.SCALE = 4
        self.N_CHANNELS = 3
        self.BALL_IDX = 0


        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
        self._attribute_tensor = None
        self._iter = None

        self._color_map = {
            self.BALL_IDX: (0, 255, 0),  # pure green for easier detection
            self.PADDLE_IDX: CLASSIC_PADDLE_COLOR,
            self.WALL_IDX: CLASSIC_WALL_COLOR,
            self.BRICK_IDX: CLASSIC_BRICK_COLORS[0],
            self.M: CLASSIC_BACKGROUND_COLOR
        }

    def set_attribute_tensor(self, attribute_tensor, iter):
        self._attribute_tensor = attribute_tensor
        self._iter = iter

    def _gen_image(self, t, check_correctness=False):
        row_indices, col_indices = np.where(self._attribute_tensor[t])
        assert row_indices.size == np.unique(row_indices).size, \
            'CONFLICT: several bits per pixel'

        if check_correctness and t >= self.FRAME_STACK_SIZE:
            self._check_correctness(col_indices)

        # print('row_indices', row_indices.shape)
        # print('col_indices', col_indices.shape)

        flat_pixels = np.full((self.N, self.N_CHANNELS), self.M, dtype=np.uint8)

        colors = np.array([self._color_map[col_idx] for col_idx in col_indices])

        # print('colors', colors.shape)
        # print('target', flat_pixels[row_indices, :].shape)
        # print()
        if row_indices.size > 0:
            flat_pixels[row_indices, :] = colors

        pixels = flat_pixels.reshape((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.N_CHANNELS))
        return pixels

    def gen_images(self, check_correctness=False):
        for t in range(self._attribute_tensor.shape[0]):
            pixels = self._gen_image(t, check_correctness)
            image = Image.fromarray(pixels)
            image = image.resize((self.SCREEN_WIDTH * self.SCALE, self.SCREEN_HEIGHT * self.SCALE))
            image.save('./inner_images/iter:{}_t:{}.png'.format(self._iter, t))

    def _check_correctness(self, col_indices):
        n_predicted_balls = np.count_nonzero(col_indices == self.BALL_IDX)
        if n_predicted_balls == 0:
            print('The ball has **NOT** been predicted.')
        elif n_predicted_balls > 1:
            print('The ball has been predicted **MULTIPLE** times.')
        else:
            print('The ball has been predicted successfully')
