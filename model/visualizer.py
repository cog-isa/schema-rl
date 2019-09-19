from .constants import Constants

from environment.schema_games.breakout.constants import \
    CLASSIC_BACKGROUND_COLOR, CLASSIC_BALL_COLOR, CLASSIC_BRICK_COLORS, \
    CLASSIC_PADDLE_COLOR, CLASSIC_WALL_COLOR

class Visualizer(Constants):
    def __init__(self):
        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
        self._attribute_tensor = None

    def set_attribute_tensor(self, attribute_tensor):
        self._attribute_tensor = attribute_tensor

    def gen_images(self):
        pass

    def check_correctness(self):
        shape = self._attribute_tensor.shape
        for t in range(shape[0]):
            matrix_sum = self._attribute_tensor[t].sum()
            if matrix_sum != self.N:
                print('t: {}, sum: {}, true_N: {}'.format(t, matrix_sum, self.N))
