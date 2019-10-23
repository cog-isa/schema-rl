import os
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

        # colors
        self._color_map = {
            self.BALL_IDX: (0, 255, 0),  # pure green for easier detection
            self.PADDLE_IDX: CLASSIC_PADDLE_COLOR,  # red-like
            self.WALL_IDX: CLASSIC_WALL_COLOR,  # gray-like
            self.BRICK_IDX: CLASSIC_BRICK_COLORS[0],  # dark-blue-like
            self.BACKGROUND_IDX: CLASSIC_BACKGROUND_COLOR  # pure black
        }
        self.SEPARATOR_COLOR = (255, 255, 255)  # pure white
        self.BAD_ENTITY_COLOR = (255, 0, 0)  # pure red

        self.BACKGROUND_COLOR = CLASSIC_BACKGROUND_COLOR
        self.INACTIVE_ACTION_SLOT_COLOR = (255, 255, 255)
        self.ACTIVE_ACTION_SLOT_COLOR = (0, 255, 0)

    def set_params(self, attribute_tensor, iter):
        self._attribute_tensor = attribute_tensor
        self._iter = iter

    def _check_entities_for_correctness(self, entities):
        _, col_indices = np.where(entities)
        n_predicted_balls = np.count_nonzero(col_indices == self.BALL_IDX)
        if n_predicted_balls == 0:
            print('BAD_BALL: zero balls exist.')
        elif n_predicted_balls > 1:
            print('BAD_BALL: multiple balls exist.')
        else:
            print('OKAY: Only one ball exists.')

    def _convert_entities_to_pixels(self, entities):
        """
        :param entities: ndarray (n_entities x M)
        :return: flat_pixels: ndarray (n_entities, N_CHANNELS)
        """
        n_entities, _ = entities.shape
        row_indices, col_indices = np.where(entities)

        unique, unique_index, unique_counts = np.unique(row_indices, return_index=True, return_counts=True)
        duplicate_indices = unique[unique_counts > 1]

        colors = np.array([self._color_map[col_idx] if row_idx not in duplicate_indices
                           else self.BAD_ENTITY_COLOR
                           for row_idx, col_idx in zip(unique, col_indices[unique_index])])

        if duplicate_indices.size:
            print('BAD_ENTITY (several bits per pixel): {} conflicts'.format(duplicate_indices.size))
            for idx in duplicate_indices:
                print('idx: {}, entity: {}'.format(idx, entities[idx]))
            print()
            # raise AssertionError

        flat_pixels = np.full((n_entities, self.N_CHANNELS), self.BACKGROUND_IDX, dtype=np.uint8)
        if colors.size:
            flat_pixels[unique, :] = colors

        return flat_pixels

    def visualize_entities(self, entities, image_path):
        """
        :param entities: ndarray (n_entities x M)
        """
        flat_pixels = self._convert_entities_to_pixels(entities)
        pixmap = flat_pixels.reshape((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.N_CHANNELS))
        image = Image.fromarray(pixmap)
        image = image.resize((self.SCREEN_WIDTH * self.STATE_SCALE,
                              self.SCREEN_HEIGHT * self.STATE_SCALE))
        image.save(image_path)

    def visualize_predicted_entities(self, check_correctness=False):
        for t in range(self._attribute_tensor.shape[0]):
            if check_correctness:
                self._check_entities_for_correctness(self._attribute_tensor[t])

            dir_name = './inner_images'
            file_name = 'iter_{}__t_{}.png'.format(self._iter, t)
            image_path = os.path.join(dir_name, file_name)
            self.visualize_entities(self._attribute_tensor[t], image_path)

# ------------- SCHEMA VISUALIZING ------------- #

    def _parse_schema_vector(self, vec):
        """
        :param vec: schema vector ((Fss * MR) + A)
        :return: tuple (entities: ndarray, actions: ndarray)
        """
        actions = vec[-self.ACTION_SPACE_DIM:]
        size = vec.size - actions.size

        frame_vectors = np.split(vec[:size], self.FRAME_STACK_SIZE)

        entities_stack = []
        for frame_vec in frame_vectors:
            central_entity = frame_vec[:self.M]
            ne_entities = frame_vec[self.M:]

            split = ne_entities.size // 2
            entities = np.concatenate(
                (ne_entities[:split], central_entity, ne_entities[split:])
            ).reshape(
                (self.NEIGHBORS_NUM + 1, self.M)
            )
            entities_stack.append(entities)

        active_actions = actions.nonzero()[0]
        return entities_stack, active_actions

    def _gen_schema_activation_pattern(self, vec):
        entities_stack, active_actions = self._parse_schema_vector(vec)

        pixmaps = []
        for entities in entities_stack:
            flat_pixels = self._convert_entities_to_pixels(entities)
            pixmap = flat_pixels.reshape((self.FILTER_SIZE, self.FILTER_SIZE, self.N_CHANNELS))
            pixmaps.append(pixmap)

        # taking separator's width = 1, color = 'white'
        v_separator = np.empty((self.FILTER_SIZE, 1, self.N_CHANNELS), dtype=np.uint8)
        v_separator[:, :] = self.SEPARATOR_COLOR
        concat_pixmap = np.hstack(
            (pixmaps[0], v_separator, pixmaps[1])
        )

        h_separator = np.empty((1, 2 * self.FILTER_SIZE + 1, self.N_CHANNELS), dtype=np.uint8)
        h_separator[:, :] = self.SEPARATOR_COLOR

        # adding actions indicator
        assert self.ACTION_SPACE_DIM == 3
        action_slots_indices = np.array([self.FILTER_SIZE + offset for offset in (-2, 0, 2)])
        active_slots_indices = action_slots_indices[active_actions]

        actions_indicator = np.empty((3, 2 * self.FILTER_SIZE + 1, self.N_CHANNELS), dtype=np.uint8)
        actions_indicator[:, :] = self.BACKGROUND_COLOR
        actions_indicator[1, action_slots_indices] = self.INACTIVE_ACTION_SLOT_COLOR
        actions_indicator[1, active_slots_indices] = self.ACTIVE_ACTION_SLOT_COLOR

        concat_pixmap = np.vstack((concat_pixmap, h_separator, actions_indicator))
        return concat_pixmap

    def visualize_schemas(self, W):
        file = open('./schema_images/metadata__iter_{}'.format(self._iter), 'wt')
        for attribute_idx, w in enumerate(W):
            s = 'attribute_idx: {}\n'.format(attribute_idx)
            file.write(s)
            for vec_idx, vec in enumerate(w.T):
                s = 4 * ' ' + 'vec_idx: {}\n'.format(vec_idx)
                file.write(s)

                pixmap = self._gen_schema_activation_pattern(vec)
                n_rows, n_cols, _ = pixmap.shape

                image = Image.fromarray(pixmap)
                image = image.resize((n_cols * self.SCHEMA_SCALE,
                                      n_rows * self.SCHEMA_SCALE))
                image.save('./schema_images/iter_{}__attr_{}__vec_{}.png'.format(
                    self._iter, attribute_idx, vec_idx
                ))

                file.write(8 * ' ' + str(vec.astype(int)) + '\n')
        file.close()
