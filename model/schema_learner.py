from collections import namedtuple
import os
import numpy as np
import mip.model as mip
from model.constants import Constants
from model.visualizer import Visualizer


class MipModel(Constants):
    """
    instantiated for single attr_idx
    """
    MAX_OPT_SECONDS = 60

    def __init__(self):
        self._model = mip.Model(mip.MINIMIZE, solver_name=mip.CBC)
        self._model.verbose = 0
        self._model.threads = self.N_LEARNING_THREADS

        self._w = [self._model.add_var(var_type='B') for _ in range(self.SCHEMA_VEC_SIZE)]
        self._constraints_buff = np.empty(0, dtype=object)

    def add_to_constraints_buff(self, batch, unique_idx):
        augmented_entities, target = batch
        batch_size = augmented_entities.shape[0]

        new_constraints = np.empty(batch_size, dtype=object)
        lin_combs = (1 - augmented_entities) @ self._w
        new_constraints[~target] = [lc >= 1 for lc in lin_combs[~target]]
        new_constraints[target] = [lc == 0 for lc in lin_combs[target]]

        concat_constraints = np.concatenate((self._constraints_buff, new_constraints), axis=0)
        self._constraints_buff = concat_constraints[unique_idx]

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


class GreedySchemaLearner(Constants):
    Batch = namedtuple('Batch', ['x', 'y'])

    def __init__(self):
        self._W = [np.ones((self.SCHEMA_VEC_SIZE, self.L), dtype=bool)
                   for _ in range(self.N_PREDICTABLE_ATTRIBUTES)]
        self._n_actual_schemas = np.ones(self.N_PREDICTABLE_ATTRIBUTES, dtype=np.int)

        self._buff = []
        self._replay = self.Batch(np.empty((0, self.SCHEMA_VEC_SIZE), dtype=bool),
                                  np.empty((0, self.N_PREDICTABLE_ATTRIBUTES), dtype=bool))

        self._mip_models = [MipModel() for _ in range(self.N_PREDICTABLE_ATTRIBUTES)]
        self._solved = []

        self._curr_iter = None
        self._visualizer = Visualizer(None, None, None)

    def set_curr_iter(self, curr_iter):
        self._curr_iter = curr_iter
        self._visualizer.set_iter(curr_iter)

    def take_batch(self, batch):
        if batch.x.size:
            x, y = self._handle_duplicates(batch.x, batch.y)
            filtered_batch = self.Batch(x, y)
            self._buff.append(filtered_batch)

    def _get_buff_batch(self):
        out = None
        if self._buff:
            x, y = zip(*self._buff)
            augmented_entities = np.concatenate(x, axis=0)
            targets = np.concatenate(y, axis=0)
            augmented_entities, targets = self._handle_duplicates(augmented_entities, targets)
            out = self.Batch(augmented_entities, targets)
        return out

    def _add_to_replay_and_constraints_buff(self, batch):
        replay_size = len(self._replay.x)

        x = np.concatenate((self._replay.x, batch.x), axis=0)
        y = np.concatenate((self._replay.y, batch.y), axis=0)
        x, y, unique_idx = self._handle_duplicates(x, y, return_index=True)

        self._replay = self.Batch(x, y)

        # find non-duplicate indices in new batch
        new_batch_mask = unique_idx >= replay_size
        new_non_duplicate_indices = unique_idx[new_batch_mask] - replay_size

        # find indices that will index constraints_buff + new_batch_unique synchronously with replay
        constraints_unique_idx = unique_idx.copy()
        constraints_unique_idx[new_batch_mask] = replay_size + np.arange(len(new_non_duplicate_indices))

        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            attr_batch = self.Batch(batch.x[new_non_duplicate_indices],
                                    batch.y[new_non_duplicate_indices, attr_idx])
            self._mip_models[attr_idx].add_to_constraints_buff(attr_batch, constraints_unique_idx)

    def _get_replay_batch(self):
        if self._replay.x.size:
            out = self._replay
        else:
            out = None
        return out

    def _predict_attribute(self, augmented_entities, attr_idx):
        assert augmented_entities.dtype == bool

        n_schemas = self._n_actual_schemas[attr_idx]
        W = self._W[attr_idx][:, :n_schemas]
        attribute_prediction = ~(~augmented_entities @ W)
        return attribute_prediction

    def _add_schema_vec(self, attr_idx, schema_vec):
        # write vector to the free idx
        vec_idx = self._n_actual_schemas[attr_idx]
        self._W[attr_idx][:, vec_idx] = schema_vec

        # maintain free idx
        self._n_actual_schemas[attr_idx] += 1

    def _delete_schema_vecs(self, attr_idx, vec_indices):
        W = self._W[attr_idx]

        assert vec_indices.ndim == 1
        n_schemas_deleted = len(vec_indices)

        if n_schemas_deleted:
            W = np.delete(W, vec_indices, axis=1)

            padding = np.ones((self.SCHEMA_VEC_SIZE, n_schemas_deleted), dtype=bool)
            W = np.hstack((W, padding))

            # mutate W, maintain free idx
            self._W[attr_idx] = W
            self._n_actual_schemas[attr_idx] -= n_schemas_deleted

        return n_schemas_deleted

    """
    def _solve_lp(self, objective_coefficients, augmented_entities, target, is_simplify=False):
        print('!!! RELAXED TASK SOLVING CALLED !!!')
        A_ub = augmented_entities[target == 0] - 1
        b_ub = np.full(len(A_ub), -1, dtype=np.int)
        A_eq = 1 - np.array(self._solved)
        b_eq = np.zeros(len(A_eq))
        bounds = (0, 1)

        opt_res = linprog(objective_coefficients,
                          A_ub=A_ub,
                          b_ub=b_ub,
                          A_eq=A_eq,
                          b_eq=b_eq,
                          bounds=bounds,
                          options={
                              'maxiter': self.MAX_ITER,
                              'disp': True,
                              'tol': self.LEARNING_SCHEMA_TOLERANCE
                          })
        new_schema_vector = opt_res.x if opt_res.success else None

        if is_simplify:
            print('SIMPLIFY', end=' ')

        if opt_res.success:
            print('linprog returned SUCCESS')
        else:
            print('linprog returned FAULT')
        print('total iterations: {}'.format(opt_res.nit))
        print('message: {}'.format(opt_res.message))

        return new_schema_vector
    """

    def _find_cluster(self, zp_pl_mask, zp_nl_mask, augmented_entities, target, attr_idx):
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

        # sample one entry and add it's idx to 'solved'
        zp_pl_indices = np.nonzero(zp_pl_mask)[0]
        idx = np.random.choice(zp_pl_indices)
        self._solved.append(idx)

        # resample candidates
        zp_pl_mask[idx] = False
        zp_pl_indices = np.nonzero(zp_pl_mask)[0]
        candidates = augmented_entities[zp_pl_mask]

        # solve LP
        objective_coefficients = list((1 - candidates).sum(axis=0))
        new_schema_vector = self._mip_models[attr_idx].optimize(objective_coefficients, zp_nl_mask, self._solved)

        if new_schema_vector is None:
            print('!!! Cannot find cluster !!!')
            return None

        # add all samples that are solved by just learned schema vector
        if candidates.size:
            new_predicted_attribute = (1 - candidates) @ new_schema_vector
            cluster_members_mask = np.isclose(new_predicted_attribute, 0, rtol=0, atol=self.ADDING_SCHEMA_TOLERANCE)
            n_new_members = np.count_nonzero(cluster_members_mask)

            if n_new_members:
                print('Also added to solved: {}'.format(n_new_members))
                self._solved.extend(zp_pl_indices[cluster_members_mask])

        return new_schema_vector

    def _simplify_schema(self, zp_nl_mask, schema_vector, augmented_entities, target, attr_idx):
        objective_coefficients = [1] * len(schema_vector)

        new_schema_vector = self._mip_models[attr_idx].optimize(objective_coefficients, zp_nl_mask, self._solved)
        assert new_schema_vector is not None

        return new_schema_vector

    def _binarize_schema(self, schema_vector):
        # prediction = (1 - self._solved[0]) @ (schema_vector > 0.5)
        # print('AFTER BINARIZING. Prediction on constrained vector: {}'.format(prediction))

        threshold = 0.5
        """
        print(np.isclose(schema_vector[schema_vector > threshold], 1, rtol=0, atol=1e-3).sum() / (schema_vector > threshold).sum())
        print(np.isclose(schema_vector[schema_vector < threshold], 0, rtol=0, atol=1e-3).sum() / (schema_vector < threshold).sum())
        print('---------')
        print(schema_vector[schema_vector > threshold].mean())
        print(schema_vector[schema_vector < threshold].mean())
        print('--------')
        print((schema_vector > threshold).sum())
        print((schema_vector < threshold).sum())
        print(schema_vector)
        """

        return schema_vector > threshold

    def _generate_new_schema(self, augmented_entities, targets, attr_idx):
        # make prediction on all replay
        predicted_attribute = self._predict_attribute(augmented_entities, attr_idx)

        augmented_entities = augmented_entities.astype(np.int, copy=False)
        target = targets[:, attr_idx].astype(np.int, copy=False)

        # sample only entries with zero-prediction
        zp_mask = ~predicted_attribute.any(axis=1)
        pl_mask = target == 1
        # pos and neg labels' masks
        zp_pl_mask = zp_mask & pl_mask
        zp_nl_mask = zp_mask & ~pl_mask

        new_schema_vector = self._find_cluster(zp_pl_mask, zp_nl_mask,
                                               augmented_entities, target, attr_idx)
        if new_schema_vector is None:
            return None

        new_schema_vector = self._simplify_schema(zp_nl_mask, new_schema_vector,
                                                  augmented_entities, target, attr_idx)
        if new_schema_vector is None:
            print('!!! Cannot simplify !!!')
            return None

        new_schema_vector = self._binarize_schema(new_schema_vector)

        self._solved.clear()

        return new_schema_vector

    def _handle_duplicates(self, augmented_entities, target, return_index=False):
        samples, idx = np.unique(augmented_entities, axis=0, return_index=True)
        out = [samples, target[idx, :]]
        if return_index:
            out.append(idx)
        return tuple(out)

    def _delete_incorrect_schemas(self, batch):
        augmented_entities, targets = batch
        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            attr_prediction = self._predict_attribute(augmented_entities, attr_idx)

            # false positive predictions
            mispredicted_samples_mask = attr_prediction.any(axis=1) & ~targets[:, attr_idx]

            incorrect_schemas_mask = attr_prediction[mispredicted_samples_mask, :].any(axis=0)
            incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]

            n_schemas_deleted = self._delete_schema_vecs(attr_idx, incorrect_schemas_indices)

            if n_schemas_deleted:
                print('Deleted incorrect schemas: {} of {}'.format(
                    n_schemas_deleted, self.ENTITY_NAMES[attr_idx]))

    def dump_weights(self, learned_W):
        dir_name = 'dump'
        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            file_name = 'w_{}.pkl'.format(attr_idx)
            path = os.path.join(dir_name, file_name)
            learned_W[attr_idx].dump(path)

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

        augmented_entities, targets = replay_batch

        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            #if attr_idx != self.BALL_IDX:
            #    continue

            while self._n_actual_schemas[attr_idx] < self.L:

                #print('predicted attribute vector: {}'.format(predicted_attribute))
                #print('predicted index of ball: {}'.format(np.nonzero(predicted_attribute == 1)))
                #prediction = ~(~augmented_entities.astype(bool) @ self._W[0][:, 1])
                #prediction = np.nonzero(prediction)
                #print('Prediction ball JUST AFTER: {}'.format(prediction))
                #print('THAT VECTOR::: {}'.format(self._W[0][:, 1]))

                new_schema_vec = self._generate_new_schema(augmented_entities, targets, attr_idx)
                if new_schema_vec is None:
                    break

                self._add_schema_vec(attr_idx, new_schema_vec)

        if self.VISUALIZE_SCHEMAS:
            learned_W = [W[:, ~np.all(W, axis=0)] for W in self._W]
            self._visualizer.visualize_schemas(learned_W, None)
            self.dump_weights(learned_W)

        if self.VISUALIZE_REPLAY_BUFFER:
            self._visualizer.visualize_replay_buffer(self._replay)
