import os
import numpy as np
from PIL import Image
from .constants import Constants
from .graph_utils import Attribute, Action


# colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PURPLE = (128, 0, 128)

BACKGROUND_COLOR = WHITE

BALL_COLOR = GREEN
WALL_COLOR = (142, 142, 142)
BRICK_COLOR = (66, 72, 200)
PADDLE_COLOR = (200, 72, 73)
VOID_COLOR = BLACK

BAD_ENTITY_COLOR = PURPLE

PATTERN_SEPARATOR_COLOR = (98, 234, 223)  # light-blue
INACTIVE_ACTION_SLOT_COLOR = WHITE
ACTIVE_ACTION_SLOT_COLOR = RED


class Visualizer(Constants):
    def __init__(self, tensor_handler, planner, attribute_nodes):
        self.VISUALIZATION_DIR_NAME = './visualization'
        self.ATTRIBUTE_SCHEMAS_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'attribute_schemas')
        self.REWARD_SCHEMAS_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'reward_schemas')
        self.ENTITIES_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'entities')
        self.BACKTRACKING_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'backtracking')
        self.STATE_DIR_NAME = os.path.join(self.VISUALIZATION_DIR_NAME, 'state')

        self.ITER_PADDING_LENGTH = 8
        self.TIME_STEP_PADDING_LENGTH = len(str(self.T))

        self.N_CHANNELS = 3
        self.STATE_SCALE = 4
        self.SCHEMA_SCALE = 128

        # need other objects' internal structure for visualizing purposes
        self._tensor_handler = tensor_handler
        self._planner = planner
        self._attribute_nodes = attribute_nodes

        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
        if tensor_handler is not None:
            self._attribute_tensor = self._tensor_handler.get_attribute_tensor()
        self._iter = None

        self._color_map = {
            self.BALL_IDX: BALL_COLOR,
            self.PADDLE_IDX: PADDLE_COLOR,  # red-like
            self.WALL_IDX: WALL_COLOR,  # gray-like
            self.BRICK_IDX: BRICK_COLOR,  # dark-blue-like
            self.VOID_IDX: VOID_COLOR
        }

    def set_iter(self, iter):
        self._iter = iter

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
            pass
            """
            print('BAD_ENTITY (several bits per pixel): {} conflicts'.format(duplicate_indices.size))
            for idx in duplicate_indices:
                print('idx: {}, entity: {}'.format(idx, entities[idx]))
            print()
            """
            # raise AssertionError

        flat_pixels = np.empty((n_entities, self.N_CHANNELS), dtype=np.uint8)
        flat_pixels[:] = BACKGROUND_COLOR
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
        for t in range(self.TIME_SIZE):
            if check_correctness:
                self._tensor_handler.check_entities_for_correctness(t)

            file_name = 'iter_{:0{ipl}d}__t_{:0{tspl}d}.png'.format(
                self._iter, t, ipl=self.ITER_PADDING_LENGTH, tspl=self.TIME_STEP_PADDING_LENGTH)
            image_path = os.path.join(self.ENTITIES_DIR_NAME, file_name)
            self.visualize_entities(self._attribute_tensor[t], image_path)

    def visualize_env_state(self, state):
        file_name = 'iter_{:0{ipl}d}.png'.format(
            self._iter, ipl=self.ITER_PADDING_LENGTH)
        image_path = os.path.join(self.STATE_DIR_NAME, file_name)
        self.visualize_entities(state, image_path)

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
        v_separator[:, :] = PATTERN_SEPARATOR_COLOR
        concat_pixmap = np.hstack(
            (pixmaps[0], v_separator, pixmaps[1])
        )

        h_separator = np.empty((1, 2 * self.FILTER_SIZE + 1, self.N_CHANNELS), dtype=np.uint8)
        h_separator[:, :] = PATTERN_SEPARATOR_COLOR

        # adding actions indicator
        offsets = (-2, 0, 2) if self.ACTION_SPACE_DIM == 3 else (-1, 1)
        action_slots_indices = np.array([self.FILTER_SIZE + offset for offset in offsets])
        activated_slots_indices = action_slots_indices[active_actions]

        actions_indicator = np.empty((3, 2 * self.FILTER_SIZE + 1, self.N_CHANNELS), dtype=np.uint8)
        actions_indicator[:, :] = PATTERN_SEPARATOR_COLOR
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
                file_name = 'iter_{:0{ipl}}__{}__vec_{:0{ipl}}.png'.format(
                    self._iter, self.ENTITY_NAMES[attribute_idx], vec_idx, ipl=self.ITER_PADDING_LENGTH)
                path = os.path.join(self.ATTRIBUTE_SCHEMAS_DIR_NAME, file_name)
                self._save_schema_image(vec, path)

        # reward schemas
        for reward_type, r in enumerate(R):
            for vec_idx, vec in enumerate(r.T):
                file_name = 'iter_{:0{ipl}}__{}__vec_{:0{ipl}}.png'.format(
                    self._iter, self.REWARD_NAMES[reward_type], vec_idx, ipl=self.ITER_PADDING_LENGTH)
                path = os.path.join(self.REWARD_SCHEMAS_DIR_NAME, file_name)
                self._save_schema_image(vec, path)

    # ------------- VISUALIZING BACKTRACKING -------------- #
    def find_connected_component_triplets(self, node):
        if node.activating_schema is None:
            return None
        triplets = []
        for precondition in node.activating_schema.attribute_preconditions:
            t = precondition.t
            i = precondition.entity_idx
            j = precondition.attribute_idx
            triplet = (t, i, j)
            triplets.append(triplet)

            child_triplets = self.find_connected_component_triplets(precondition)
            if child_triplets is not None:
                triplets.extend(child_triplets)
        return triplets

    def apply_triplets_to_base_state(self, triplets):
        base_state = self._attribute_tensor[self.FRAME_STACK_SIZE - 1, :, :].copy()
        for t, i, j in triplets:
            base_state[i, :] = False
            base_state[i, j] = True
        return base_state

    def visualize_node_backtracking(self, reward_node, image_path, partial_triplets):
        if partial_triplets is not None:
            triplets = partial_triplets[reward_node]
        else:
            triplets = self.find_connected_component_triplets(reward_node)
        entities = self.apply_triplets_to_base_state(triplets)
        self.visualize_entities(entities, image_path)

    def visualize_backtracking(self, target_reward_nodes, partial_triplets):
        for idx, reward_node in enumerate(target_reward_nodes):
            file_name = 'iter_{:0{ipl}d}__node_{}.png'.format(
                self._iter, idx, ipl=self.ITER_PADDING_LENGTH)
            image_path = os.path.join(self.BACKTRACKING_DIR_NAME, file_name)
            self.visualize_node_backtracking(reward_node, image_path, partial_triplets)

# -------------- LOGGING BACKTRACKING --------------- #
    def write_block(self, block, file, indent_size=0):
        if indent_size != 0:
            block = [line + (indent_size * ' ') for line in block]
        file.write('\n'.join(block) + '\n')

    def log_precondition_node(self, node, file):
        """
        expecting only attribute or action nodes
        """
        assert type(node) in (Attribute, Action)

        if type(node) is Attribute:
            block = [
                'type: {}'.format(type(node)),
                't: {}'.format(node.t),
                'is_reachable: {}'.format(node.is_reachable),
                'entity_idx: {}'.format(node.entity_idx),
                'attribute_idx: {}'.format(node.attribute_idx)
            ]
        else:
            block = [
                'type: {}'.format(type(node)),
                't: {}'.format(node.t),
                'idx: {}'.format(node.idx),
            ]
        self.write_block(block, file, indent_size=4)
        self.write_block([''], file)

    def log_schema_preconditions(self, schema_idx, schema, file):
        block = [
            '#{} schema:'.format(schema_idx),
            't: {}'.format(schema.t),
            'is_reachable: {}'.format(schema.is_reachable),
            '',
            'attribute preconditions: {}'.format(len(schema.attribute_preconditions))
        ]
        self.write_block(block, file)
        for attribute_node in schema.attribute_preconditions:
            self.log_precondition_node(attribute_node, file)

        block = ['action preconditions: {}'.format(len(schema.action_preconditions))]
        self.write_block(block, file)

        for action_node in schema.action_preconditions:
            self.log_precondition_node(action_node, file)

    def log_node_with_schemas(self, node, file):
        block = [
            '----------------',
            'NODE of type {}'.format(type(node)),
            't: {}'.format(node.t),
            'entity_idx: {}'.format(node.entity_idx),
            'attribute_idx: {}'.format(node.attribute_idx),
            'is_reachable: {}'.format(node.is_reachable),
            'activating_schema: {}'.format(node.activating_schema),
            '---',
            'n_schemas: {}'.format(len(node.schemas)),
            '---'
        ]
        self.write_block(block, file)
        for schema_idx, schema in enumerate(node.schemas):
            self.log_schema_preconditions(schema_idx, schema, file)

    def log_balls_at_backtracking(self, reward_node):
        for t in range(self.TIME_SIZE):
            ball_entity_idx = self._tensor_handler.get_ball_entity_idx(t)
            if ball_entity_idx is None:
                continue

            ball_node = self._attribute_nodes[t, ball_entity_idx, self.BALL_IDX]

            file_name = 'iter_{}__ball_node_at_time_{}'.format(self._iter, t)
            logfile_path = os.path.join(self.BACKTRACKING_DIR_NAME, file_name)
            with open(logfile_path, 'wt') as file:
                self.log_node_with_schemas(ball_node, file)


