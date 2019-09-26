import numpy as np
from PIL import Image

from .constants import Constants

from environment.schema_games.breakout.constants import \
    CLASSIC_BACKGROUND_COLOR, CLASSIC_BALL_COLOR, CLASSIC_BRICK_COLORS, \
    CLASSIC_PADDLE_COLOR, CLASSIC_WALL_COLOR


class Visualizer(Constants):
    def __init__(self, W):
        self.STATE_SCALE = 4
        self.SCHEMA_SCALE = 128
        self.N_CHANNELS = 3

        self.BACKGROUND_IDX = self.M

        self._W = W
        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
        self._attribute_tensor = None
        self._iter = None

        self._color_map = {
            self.BALL_IDX: (0, 255, 0),  # pure green for easier detection
            self.PADDLE_IDX: CLASSIC_PADDLE_COLOR,
            self.WALL_IDX: CLASSIC_WALL_COLOR,
            self.BRICK_IDX: CLASSIC_BRICK_COLORS[0],
            self.BACKGROUND_IDX: CLASSIC_BACKGROUND_COLOR
        }

        if self.VISUALIZE_SCHEMAS:
            self.visualize_schemas(self._W)

    def set_attribute_tensor(self, attribute_tensor, iter):
        self._attribute_tensor = attribute_tensor
        self._iter = iter

    def _convert_entities_to_pixels(self, entities):
        """
        :param entities: ndarray (n_entities x M)
        :return: flat_pixels: ndarray (n_entities, N_CHANNELS)
        """
        n_entities, _ = entities.shape
        row_indices, col_indices = np.where(entities)
        assert row_indices.size == np.unique(row_indices).size, \
            'CONFLICT: several bits per pixel'
        colors = np.array([self._color_map[col_idx] for col_idx in col_indices])

        flat_pixels = np.full((n_entities, self.N_CHANNELS), self.BACKGROUND_IDX, dtype=np.uint8)
        if colors.size:
            flat_pixels[row_indices, :] = colors

        return flat_pixels

    def _gen_state_pixmap(self, t):
        flat_pixels = self._convert_entities_to_pixels(self._attribute_tensor[t])
        pixmap = flat_pixels.reshape((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.N_CHANNELS))
        return pixmap

    def visualize_inner_state(self, check_correctness=False):
        for t in range(self._attribute_tensor.shape[0]):
            if check_correctness:
                self._check_correctness(self._attribute_tensor[t])

            pixmap = self._gen_state_pixmap(t)
            image = Image.fromarray(pixmap)
            image = image.resize((self.SCREEN_WIDTH * self.STATE_SCALE,
                                  self.SCREEN_HEIGHT * self.STATE_SCALE))
            image.save('./inner_images/iter:{}_t:{}.png'.format(self._iter, t))

    def _check_correctness(self, entities):
        _, col_indices = np.where(entities)
        n_predicted_balls = np.count_nonzero(col_indices == self.BALL_IDX)
        if n_predicted_balls == 0:
            print('The ball has **NOT** been predicted.')
        elif n_predicted_balls > 1:
            print('The ball has been predicted **MULTIPLE** times.')
        else:
            print('The ball has been predicted successfully')

    def _gen_schema_pixmap(self, vec):
        """
        :param vec: schema vector ((Fss * MR) + A)
        :return: tuple (pixmap: ndarray, actions: ndarray)
        """
        actions = vec[-self.ACTION_SPACE_DIM:]
        size = vec.size - actions.size

        frame_vectors = np.split(vec[:size], self.FRAME_STACK_SIZE)

        pixmaps = []
        dim = 2 * self.NEIGHBORHOOD_RADIUS + 1
        for frame_vec in frame_vectors:
            central_entity = frame_vec[:self.M]
            ne_entities = frame_vec[self.M:]

            split = ne_entities.size // 2
            entities = np.concatenate(
                (ne_entities[:split], central_entity, ne_entities[split:])
            ).reshape(
                (self.NEIGHBORS_NUM + 1, self.M)
            )

            flat_pixels = self._convert_entities_to_pixels(entities)
            pixmap = flat_pixels.reshape((dim, dim, self.N_CHANNELS))
            pixmaps.append(pixmap)

        # taking separator width = 2
        concat_pixmap = np.hstack((
            pixmaps[0],
            np.full((dim, 2, self.N_CHANNELS), CLASSIC_BACKGROUND_COLOR, dtype=np.uint8),
            pixmaps[1]
        ))
        return concat_pixmap, actions

    def visualize_schemas(self, W):
        for attribute_idx, w in enumerate(W):
            for vec_idx, vec in enumerate(w.T):
                pixmap, actions = self._gen_schema_pixmap(vec)
                n_rows, n_cols, _ = pixmap.shape

                image = Image.fromarray(pixmap)
                image = image.resize((n_cols * self.SCHEMA_SCALE,
                                      n_rows * self.SCHEMA_SCALE))
                image.save('./schema_images/iter_{}__attr_{}__vec_{}.png'.format(
                    self._iter, attribute_idx, vec_idx
                ))




































