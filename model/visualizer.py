import os
import numpy as np
from PIL import Image
from .constants import Constants


# colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

BACKGROUND_COLOR = (0, 0, 0)
WALL_COLOR = (142, 142, 142)
BRICK_COLOR = (66, 72, 200)
PADDLE_COLOR = (200, 72, 73)
BAD_ENTITY_COLOR = WHITE

SEPARATOR_COLOR = WHITE
INACTIVE_ACTION_SLOT_COLOR = WHITE
ACTIVE_ACTION_SLOT_COLOR = RED


class Visualizer(Constants):
    def __init__(self):
        self.VISUALIZATION_DIR_NAME = './visualization'
        self.ATTRIBUTE_SCHEMAS_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'attribute_schemas')
        self.REWARD_SCHEMAS_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'reward_schemas')
        self.ENTITIES_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'entities')

        self.N_CHANNELS = 3
        self.STATE_SCALE = 4
        self.SCHEMA_SCALE = 128

        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
        self._attribute_tensor = None
        self._iter = None

        self._color_map = {
            self.BALL_IDX: GREEN,
            self.PADDLE_IDX: PADDLE_COLOR,  # red-like
            self.WALL_IDX: WALL_COLOR,  # gray-like
            self.BRICK_IDX: BRICK_COLOR,  # dark-blue-like
            self.VOID_IDX: BACKGROUND_COLOR  # pure black
        }

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
                           else BAD_ENTITY_COLOR
                           for row_idx, col_idx in zip(unique, col_indices[unique_index])])

        if duplicate_indices.size:
            print('BAD_ENTITY (several bits per pixel): {} conflicts'.format(duplicate_indices.size))
            for idx in duplicate_indices:
                print('idx: {}, entity: {}'.format(idx, entities[idx]))
            print()
            # raise AssertionError

        flat_pixels = np.full((n_entities, self.N_CHANNELS), self.VOID_IDX, dtype=np.uint8)
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

            file_name = 'iter_{}__t_{}.png'.format(self._iter, t)
            image_path = os.path.join(self.ENTITIES_DIR_NAME, file_name)
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

    def _gen_schema_activation_pattern(self, entities_stack, active_actions):
        pixmaps = []
        for entities in entities_stack:
            flat_pixels = self._convert_entities_to_pixels(entities)
            pixmap = flat_pixels.reshape((self.FILTER_SIZE, self.FILTER_SIZE, self.N_CHANNELS))
            pixmaps.append(pixmap)

        # taking separator's width = 1, color = 'white'
        v_separator = np.empty((self.FILTER_SIZE, 1, self.N_CHANNELS), dtype=np.uint8)
        v_separator[:, :] = SEPARATOR_COLOR
        concat_pixmap = np.hstack(
            (pixmaps[0], v_separator, pixmaps[1])
        )

        h_separator = np.empty((1, 2 * self.FILTER_SIZE + 1, self.N_CHANNELS), dtype=np.uint8)
        h_separator[:, :] = SEPARATOR_COLOR

        # adding actions indicator
        offsets = (-2, 0, 2) if self.ACTION_SPACE_DIM == 3 else (-1, 1)
        action_slots_indices = np.array([self.FILTER_SIZE + offset for offset in offsets])
        activated_slots_indices = action_slots_indices[active_actions]

        actions_indicator = np.empty((3, 2 * self.FILTER_SIZE + 1, self.N_CHANNELS), dtype=np.uint8)
        actions_indicator[:, :] = BACKGROUND_COLOR
        actions_indicator[1, action_slots_indices] = INACTIVE_ACTION_SLOT_COLOR
        actions_indicator[1, activated_slots_indices] = ACTIVE_ACTION_SLOT_COLOR

        concat_pixmap = np.vstack((concat_pixmap, h_separator, actions_indicator))
        return concat_pixmap

    def _save_schema_image(self, vec, image_path):
        entities_stack, active_actions = self._parse_schema_vector(vec)
        pixmap = self._gen_schema_activation_pattern(entities_stack, active_actions)
        n_rows, n_cols, _ = pixmap.shape

        image = Image.fromarray(pixmap)
        image = image.resize((n_cols * self.SCHEMA_SCALE,
                              n_rows * self.SCHEMA_SCALE))
        image.save(image_path)

    def visualize_schemas(self, W, R):
        # attribute schemas
        for attribute_idx, w in enumerate(W):
            for vec_idx, vec in enumerate(w.T):
                file_name = 'iter_{}__{}__vec_{}.png'.format(self._iter,
                                                             self.ENTITY_NAMES[attribute_idx],
                                                             vec_idx)
                path = os.path.join(self.ATTRIBUTE_SCHEMAS_DIR_NAME, file_name)
                self._save_schema_image(vec, path)

        # reward schemas
        for reward_type, r in enumerate(R):
            for vec_idx, vec in enumerate(r.T):
                file_name = 'iter_{}__{}__vec_{}.png'.format(self._iter,
                                                             self.REWARD_NAMES[reward_type],
                                                             vec_idx)
                path = os.path.join(self.REWARD_SCHEMAS_DIR_NAME, file_name)
                self._save_schema_image(vec, path)
