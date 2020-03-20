from collections import namedtuple
import os
import numpy as np
import mip.model as mip
from model.constants import Constants as C
from model.visualizer import Visualizer


class MipModel:
    """
    instantiated for single attr_idx
    """
    MAX_OPT_SECONDS = 60

    def __init__(self):
        self._model = mip.Model(mip.MINIMIZE, solver_name=mip.CBC)
        self._model.verbose = 0
        self._model.threads = C.N_LEARNING_THREADS

        self._w = [self._model.add_var(var_type='B') for _ in range(C.SCHEMA_VEC_SIZE)]
        self._constraints_buff = np.empty(0, dtype=object)

    def add_to_constraints_buff(self, batch, unique_idx, replay_renewed_indices=None):
        augmented_entities, target = batch
        batch_size = augmented_entities.shape[0]

        new_constraints = np.empty(batch_size, dtype=object)
        lin_combs = (1 - augmented_entities) @ self._w
        new_constraints[~target] = [lc >= 1 for lc in lin_combs[~target]]
        new_constraints[target] = [lc == 0 for lc in lin_combs[target]]

        concat_constraints = np.concatenate((self._constraints_buff, new_constraints), axis=0)
        self._constraints_buff = concat_constraints[unique_idx]

        if replay_renewed_indices is not None:
            for idx in replay_renewed_indices:
                constr = self._constraints_buff[idx]
                assert constr.sense == '=', constr.sense
                assert constr.const == 0, constr.const
                constr = constr >= 1
                assert constr.sense == '>', constr.sense
                assert constr.const == -1, constr.const
                self._constraints_buff[idx] = constr

    def optimize(self, objective_coefficients, zp_nl_mask, solved):
        model = self._model

        # add objective
        model.objective = mip.xsum(x_i * w_i for x_i, w_i in zip(objective_coefficients, self._w))

        # add constraints
        model.remove([constr for constr in model.constrs])

        constraints_mask = zp_nl_mask
        constraints_mask[solved] = True

        constraints_to_add = self._constraints_buff[constraints_mask]
        for constraint in constraints_to_add:
            model.add_constr(constraint)

        # optimize
        status = model.optimize(max_seconds=self.MAX_OPT_SECONDS)

        if status == mip.OptimizationStatus.OPTIMAL:
            print('Optimal solution cost {} found'.format(
                model.objective_value))
        elif status == mip.OptimizationStatus.FEASIBLE:
            print('Sol.cost {} found, best possible: {}'.format(
                model.objective_value, model.objective_bound))
        elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
            print('No feasible solution found, lower bound is: {}'.format(
                model.objective_bound))
        else:
            print('Optimization FAILED.')

        if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
            schema_vec = np.array([v.x for v in model.vars])
        else:
            schema_vec = None
        return schema_vec


class GreedySchemaLearner:
    Batch = namedtuple('Batch', ['x', 'y', 'r'])

    def __init__(self):
        self._W = [np.ones((C.SCHEMA_VEC_SIZE, C.L), dtype=bool)
                   for _ in range(C.N_PREDICTABLE_ATTRIBUTES)]
        self._n_attr_schemas = np.ones(C.N_PREDICTABLE_ATTRIBUTES, dtype=np.int)

        self._R = np.ones((C.SCHEMA_VEC_SIZE, C.L), dtype=bool)
        self._n_reward_schemas = 1

        self._buff = []
        self._replay = self.Batch(np.empty((0, C.SCHEMA_VEC_SIZE), dtype=bool),
                                  np.empty((0, C.N_PREDICTABLE_ATTRIBUTES), dtype=bool),
                                  np.empty((0), dtype=bool))

        self._attr_mip_models = [MipModel() for _ in range(C.N_PREDICTABLE_ATTRIBUTES)]
        self._reward_mip_model = MipModel()
        self._solved = []

        self._curr_iter = None
        self._visualizer = Visualizer(None, None, None)

    def set_curr_iter(self, curr_iter):
        self._curr_iter = curr_iter
        self._visualizer.set_iter(curr_iter)

    def take_batch(self, batch):
        for part in batch:
            assert part.dtype == bool

        if batch.x.size:
            assert np.all(batch.r == batch.r[0])
            x, y, r = self._handle_duplicates(batch.x, batch.y, batch.r)
            filtered_batch = self.Batch(x, y, r)
            self._buff.append(filtered_batch)

    def _handle_duplicates(self, augmented_entities, target, rewards, return_index=False):
        samples, idx = np.unique(augmented_entities, axis=0, return_index=True)
        out = [samples, target[idx, :], rewards[idx]]
        if return_index:
            out.append(idx)
        return tuple(out)

    def _get_buff_batch(self):
        out = None
        if self._buff:
            # sort buff to keep r = 0 entries
            self._buff = sorted(self._buff, key=lambda batch: batch.r[0])

            x, y, r = zip(*self._buff)
            augmented_entities = np.concatenate(x, axis=0)
            targets = np.concatenate(y, axis=0)
            rewards = np.concatenate(r, axis=0)

            augmented_entities, targets, rewards = self._handle_duplicates(augmented_entities, targets, rewards)
            out = self.Batch(augmented_entities, targets, rewards)

            self._buff.clear()
        return out

    def _add_to_replay_and_constraints_buff(self, batch):
        batch_size = len(batch.x)
        old_replay_size = len(self._replay.x)

        # concatenate replay + batch
        x_concat = np.concatenate((self._replay.x, batch.x), axis=0)
        y_concat = np.concatenate((self._replay.y, batch.y), axis=0)
        r_concat = np.concatenate((self._replay.r, batch.r), axis=0)

        concat_size = len(x_concat)

        # remove duplicates
        x_filtered, y_filtered, r_filtered, unique_idx = self._handle_duplicates(
            x_concat, y_concat, r_concat, return_index=True)

        # check if np.unique prefer first elements
        mask_concat = np.zeros(concat_size, dtype=bool)
        mask_concat[unique_idx] = True
        assert mask_concat[:old_replay_size].all()

        self._replay = self.Batch(x_filtered, y_filtered, r_filtered)

        # find r = 0 duplicates (they can only locate in batch)
        duplicates_mask_concat = np.ones(concat_size, dtype=bool)
        duplicates_mask_concat[unique_idx] = False
        zero_reward_mask_concat = (r_concat == 0)
        reward_renew_indices = np.nonzero(duplicates_mask_concat & zero_reward_mask_concat)[0]
        assert (reward_renew_indices >= old_replay_size).all()
        samples_to_update = x_concat[reward_renew_indices]

        # update rewards to zero
        replay_indices_to_update = []
        for sample in samples_to_update:
            new_replay_indices = np.nonzero((self._replay.x == sample).all(axis=1))[0]
            assert len(new_replay_indices) == 1
            new_replay_idx = new_replay_indices[0]
            if self._replay.r[new_replay_idx] != 0:
                self._replay.r[new_replay_idx] = 0
                replay_indices_to_update.append(new_replay_idx)
        print('Nullified rewards of {} old samples.'.format(len(replay_indices_to_update)))

        # find non-duplicate indices in new batch (batch-based indexing)
        batch_mask_of_concat = unique_idx >= old_replay_size
        new_non_duplicate_indices = unique_idx[batch_mask_of_concat] - old_replay_size

        # find indices that will index constraints_buff + new_batch_unique synchronously with replay
        constraints_unique_idx = unique_idx.copy()
        constraints_unique_idx[batch_mask_of_concat] = old_replay_size + np.arange(len(new_non_duplicate_indices))

        for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
            attr_batch = (batch.x[new_non_duplicate_indices],
                          batch.y[new_non_duplicate_indices, attr_idx])
            self._attr_mip_models[attr_idx].add_to_constraints_buff(attr_batch, constraints_unique_idx)

        reward_batch = (batch.x[new_non_duplicate_indices],
                        batch.r[new_non_duplicate_indices])
        self._reward_mip_model.add_to_constraints_buff(reward_batch, constraints_unique_idx,
                                                       replay_renewed_indices=replay_indices_to_update)

    def _get_replay_batch(self):
        if self._replay.x.size:
            out = self._replay
        else:
            out = None
        return out

    def _predict_attribute(self, augmented_entities, attr_idx):
        assert augmented_entities.dtype == bool

        n_schemas = self._n_attr_schemas[attr_idx]
        W = self._W[attr_idx][:, :n_schemas]
        attribute_prediction = ~(~augmented_entities @ W)
        return attribute_prediction

    def _predict_reward(self, augmented_entities):
        assert augmented_entities.dtype == bool

        R = self._R[:, :self._n_reward_schemas]
        reward_prediction = ~(~augmented_entities @ R)
        return reward_prediction

    def _add_attr_schema_vec(self, attr_idx, schema_vec):
        vec_idx = self._n_attr_schemas[attr_idx]
        if vec_idx < C.L:
            self._W[attr_idx][:, vec_idx] = schema_vec
            self._n_attr_schemas[attr_idx] += 1

    def _add_reward_schema_vec(self, schema_vec):
        vec_idx = self._n_reward_schemas
        if vec_idx < C.L:
            self._R[:, vec_idx] = schema_vec
            self._n_reward_schemas += 1

    def _purge_matrix_columns(self, matrix, col_indices):
        n_cols_purged = len(col_indices)

        if n_cols_purged:
            col_size, _ = matrix.shape
            matrix = np.delete(matrix, col_indices, axis=1)
            padding = np.ones((col_size, n_cols_purged), dtype=bool)
            matrix = np.hstack((matrix, padding))

        return matrix, n_cols_purged

    def _delete_attr_schema_vectors(self, attr_idx, vec_indices):
        matrix, n_cols_purged = self._purge_matrix_columns(self._W[attr_idx], vec_indices)
        if n_cols_purged:
            self._W[attr_idx] = matrix
            self._n_attr_schemas[attr_idx] -= n_cols_purged
        return n_cols_purged

    def _delete_reward_schema_vectors(self, vec_indices):
        matrix, n_cols_purged = self._purge_matrix_columns(self._R, vec_indices)
        if n_cols_purged:
            self._R = matrix
            self._n_reward_schemas -= n_cols_purged
        return n_cols_purged

    def _delete_incorrect_schemas(self, batch):
        augmented_entities, targets, rewards = batch
        for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
            attr_prediction = self._predict_attribute(augmented_entities, attr_idx)

            # false positive predictions
            mispredicted_samples_mask = attr_prediction.any(axis=1) & ~targets[:, attr_idx]

            incorrect_schemas_mask = attr_prediction[mispredicted_samples_mask, :].any(axis=0)
            incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]

            assert incorrect_schemas_indices.ndim == 1

            n_schemas_deleted = self._delete_attr_schema_vectors(attr_idx, incorrect_schemas_indices)

            if n_schemas_deleted:
                print('Deleted incorrect attr schemas: {} of {}'.format(
                    n_schemas_deleted, C.ENTITY_NAMES[attr_idx]))

        reward_prediction = self._predict_reward(augmented_entities)

        # false positive predictions
        mispredicted_samples_mask = reward_prediction.any(axis=1) & ~rewards

        incorrect_schemas_mask = reward_prediction[mispredicted_samples_mask, :].any(axis=0)
        incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]

        assert incorrect_schemas_indices.ndim == 1
        n_schemas_deleted = self._delete_reward_schema_vectors(incorrect_schemas_indices)

        if n_schemas_deleted:
            print('Deleted incorrect reward schemas: {}'.format(n_schemas_deleted))

    def _find_cluster(self, zp_pl_mask, zp_nl_mask, augmented_entities, target, attr_idx, opt_model):
        """
        augmented_entities: zero-predicted only
        target: scalar vector
        """
        assert augmented_entities.dtype == np.int
        assert target.dtype == np.int

        # find all entries, that can be potentially solved (have True labels)
        candidates = augmented_entities[zp_pl_mask]

        print('finding cluster...')
        print('augmented_entities: {}'.format(augmented_entities.shape[0]))
        print('zp pos samples: {}'.format(candidates.shape[0]))

        if not candidates.size:
            return None

        zp_pl_indices = np.nonzero(zp_pl_mask)[0]

        #if not is_reward:
        # sample one entry and add it's idx to 'solved'
        idx = np.random.choice(zp_pl_indices)
        self._solved.append(idx)

        # resample candidates
        zp_pl_mask[idx] = False
        zp_pl_indices = np.nonzero(zp_pl_mask)[0]
        candidates = augmented_entities[zp_pl_mask]

        # solve LP
        objective_coefficients = (1 - candidates).sum(axis=0)
        objective_coefficients = list(objective_coefficients)

        new_schema_vector = opt_model.optimize(objective_coefficients, zp_nl_mask, self._solved)

        if new_schema_vector is None:
            print('!!! Cannot find cluster !!!')
            return None

        # add all samples that are solved by just learned schema vector
        if candidates.size:
            new_predicted_attribute = (1 - candidates) @ new_schema_vector
            cluster_members_mask = np.isclose(new_predicted_attribute, 0, rtol=0, atol=C.ADDING_SCHEMA_TOLERANCE)
            n_new_members = np.count_nonzero(cluster_members_mask)

            if n_new_members:
                print('Also added to solved: {}'.format(n_new_members))
                self._solved.extend(zp_pl_indices[cluster_members_mask])

        return new_schema_vector

    def _simplify_schema(self, zp_nl_mask, schema_vector, opt_model):
        objective_coefficients = [1] * len(schema_vector)

        new_schema_vector = opt_model.optimize(objective_coefficients, zp_nl_mask, self._solved)
        assert new_schema_vector is not None

        return new_schema_vector

    def _binarize_schema(self, schema_vector):
        threshold = 0.5
        return schema_vector > threshold

    def _generate_new_schema(self, augmented_entities, targets, attr_idx, is_reward=False):
        if not is_reward:
            target = targets[:, attr_idx].astype(np.int, copy=False)
            prediction = self._predict_attribute(augmented_entities, attr_idx)
            opt_model = self._attr_mip_models[attr_idx]
        else:
            target = targets.astype(np.int, copy=False)
            prediction = self._predict_reward(augmented_entities)
            opt_model = self._reward_mip_model

        augmented_entities = augmented_entities.astype(np.int, copy=False)

        # sample only entries with zero-prediction
        zp_mask = ~prediction.any(axis=1)
        pl_mask = target == 1
        # pos and neg labels' masks
        zp_pl_mask = zp_mask & pl_mask
        zp_nl_mask = zp_mask & ~pl_mask

        new_schema_vector = self._find_cluster(zp_pl_mask, zp_nl_mask,
                                               augmented_entities, target, attr_idx,
                                               opt_model)
        if new_schema_vector is None:
            return None

        new_schema_vector = self._simplify_schema(zp_nl_mask, new_schema_vector, opt_model)
        if new_schema_vector is None:
            print('!!! Cannot simplify !!!')
            return None

        new_schema_vector = self._binarize_schema(new_schema_vector)

        self._solved.clear()

        return new_schema_vector

    def dump_weights(self, learned_W, learned_R):
        dir_name = 'dump'
        os.makedirs(dir_name, exist_ok=True)
        for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
            file_name = 'w_{}.pkl'.format(attr_idx)
            path = os.path.join(dir_name, file_name)
            learned_W[attr_idx].dump(path)

        file_name = 'r_pos.pkl'
        path = os.path.join(dir_name, file_name)
        learned_R[0].dump(path)

    def get_weights(self):
        learned_W = [W[:, ~np.all(W, axis=0)] for W in self._W]
        if any(w.size == 0 for w in learned_W):
            learned_W = None

        learned_R = [self._R[:, ~np.all(self._R, axis=0)]]
        if learned_R[0].size == 0:
            learned_R = None
        return learned_W, learned_R

    def learn(self):
        # get full batch from buffer
        buff_batch = self._get_buff_batch()
        if buff_batch is not None:
            self._delete_incorrect_schemas(buff_batch)
            self._add_to_replay_and_constraints_buff(buff_batch)

        # get all data to learn on
        replay_batch = self._get_replay_batch()
        if replay_batch is None:
            return

        augmented_entities, targets, rewards = replay_batch

        if not C.USE_HANDCRAFTED_ATTRIBUTE_SCHEMAS:
            for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
                while self._n_attr_schemas[attr_idx] < C.L:
                    new_schema_vec = self._generate_new_schema(augmented_entities, targets, attr_idx)
                    if new_schema_vec is None:
                        break
                    self._add_attr_schema_vec(attr_idx, new_schema_vec)

        if not C.USE_HANDCRAFTED_REWARD_SCHEMAS:
            while self._n_reward_schemas < C.L:
                new_schema_vec = self._generate_new_schema(augmented_entities, rewards, None, is_reward=True)
                if new_schema_vec is None:
                    break
                self._add_reward_schema_vec(new_schema_vec)

        if C.VISUALIZE_SCHEMAS:
            learned_W = [W[:, ~np.all(W, axis=0)] for W in self._W]
            learned_R = [self._R[:, ~np.all(self._R, axis=0)]]
            self._visualizer.visualize_schemas(learned_W, learned_R)
            self.dump_weights(learned_W, learned_R)

        if C.VISUALIZE_REPLAY_BUFFER:
            self._visualizer.visualize_replay_buffer(self._replay)
