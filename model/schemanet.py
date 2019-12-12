import numpy as np
from scipy.optimize import linprog
import os
from termcolor import colored
from model.constants import Constants


class SchemaNet(Constants):
    def __init__(self, void=1, is_for_reward=False):
        self.neighbour_num = ((self.NEIGHBORHOOD_RADIUS * 2 + 1) ** 2) * self.FRAME_STACK_SIZE
        self._W = [np.zeros([self.neighbour_num * self.M + self.ACTION_SPACE_DIM, 1])[:] + 1 for i in range(self.M - 1)]
        self.solved = np.array([])
        self.void = void
        self.is_for_reward = is_for_reward

    def log(self):
        print('current net:\n', [self._W[i].shape for i in range(len(self._W))])

    # def

    def _logical_filter(self, w, i, log=False):

        # index of previous central attribute
        ind = self.M * (self.NEIGHBORHOOD_RADIUS * 2 + 1) ** 2 + i

        # if smth happened while agent stayed sill that was not due to agent's action
        if w[-self.ACTION_SPACE_DIM] and not w[ind]:
            schema = w
            schema[-self.ACTION_SPACE_DIM] = False
            if log:
                print('action deleted')
            return schema
        return w

    def _predict_attr(self, X, i):
        if len(self._W[i].shape) == 1:
            return ((X == 0) @ self._W[i]) == 0
        return (((X == 0) @ self._W[i]) == 0).any(axis=1) != 0

    def _schema_predict_attr(self, X, i):
        return ((X == 0) @ self._W[i] == 0) != 0

    def get_schema_num(self):
        if self.is_for_reward:
            return self.REWARD_SPACE_DIM
        else:
            return self.M - self.void

    def predict(self, X):
        tmp = np.array([self._predict_attr(X, i) for i in range(self.get_schema_num())])
        return np.concatenate((tmp, (tmp.sum(axis=0) == 0).reshape(1, -1)), axis=0)

    def add(self, schemas, i):
        self._W[i] = np.vstack([self._W[i].T, schemas.T]).T

    def scipy_solve_lp(self, zero_pred, c, A_ub, b_ub, A_eq, b_eq, maxiter=200):
        options = {'maxiter': maxiter, "disp": False}
        if len(zero_pred) == 0:
            return linprog(c=c, A_eq=A_eq, b_eq=b_eq, options=options).x.round(2)
        else:
            return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, options=options).x.round(2)

    def _get_not_predicted(self, X, y, i):
        ind = (self._predict_attr(X, i) != y) | (y == 0)
        return X[ind], y[ind]

    def _get_next_to_predict(self, X, y, i, log=False):
        ind = (self._predict_attr(X, i) != y) * (y == 1)

        if ind.sum() == 0:
            print(colored('FATAL ERROR', 'red'))
            return None
        result = X[ind][0]
        return result

    def _get_schema(self, X, y, i, log=True):

        zero_pred = X[y == 0]
        ones_pred = X[y == 1]

        new_ent = self._get_next_to_predict(X, y, i, log=log)
        self.solved = np.array([new_ent])
        if self.solved is None:
            return None

        c = (1 - ones_pred).sum(axis=0)
        A_eq = 1 - self.solved
        b_eq = np.zeros(self.solved.shape[0])
        A_ub = zero_pred - 1
        b_ub = np.zeros(zero_pred.shape[0]) - 1
        w = self.scipy_solve_lp(zero_pred, c, A_ub, b_ub, A_eq, b_eq)

        preds = ((X == 0) @ w) == 0
        print('expected:', (X[preds * (self._predict_attr(X, i) == 0)]).shape)
        print('needed for:', (self._predict_attr(X, i) == 0).sum(), preds.sum())
        if preds.sum() == 0:
            return None
        self.solved = np.vstack([X[preds * (self._predict_attr(X, i) == 0)]])
        print('solved for:', self.solved.shape)
        if self.solved is None:
            print('CONFLICT DATA')
            return None
        return w

    def _simplify_schema(self, X, y):

        zero_pred = X[y == 0]

        c = np.zeros(self.neighbour_num * self.M + self.ACTION_SPACE_DIM)
        A_eq = (1 - self.solved)
        b_eq = np.zeros(self.solved.shape[0])
        A_ub = (zero_pred - 1)
        b_ub = np.zeros(zero_pred.shape[0]) - 1

        return self.scipy_solve_lp(zero_pred, c, A_ub, b_ub, A_eq, b_eq)

    def _actuality_check_attr(self, X, y, i):
        if len(self._W[i].shape) == 1:
            return np.zeros(self._W[i].shape[0])
        pred = self._schema_predict_attr(X, i).T
        return (y[i] - pred) == -1

    def _check_lim_size(self):
        return np.all([schema.shape[1] < self.L for schema in self._W])

    def _remove_wrong_schemas(self, X, y):
        for i in range(self.get_schema_num()):
            if len(self._W[i].shape) == 1:
                break
            wrong_ind = self._actuality_check_attr(X, y, i).sum(axis=1)

            if (wrong_ind.sum()) != 0:
                print('outdated schema was detected for attribute', i)

            self._W[i] = (self._W[i].T[wrong_ind == 0]).T

    def fit(self, X, Y, log=True):
        tmp, ind = np.unique(X, return_index=True, axis=0)
        print('index shape', ind.shape, X.shape, Y.shape)

        X = X[ind]

        Y = (Y.T[ind]).T

        self._remove_wrong_schemas(X, Y)

        for i in (range(self.get_schema_num())):

            while self._check_lim_size():

                if (self._predict_attr(X, i) == Y[i]).all():
                    if log:
                        if i == 0:
                            print('ball check', (self._predict_attr(X, i) == 1).any(), (Y[i] == 1).any())

                        print('all attrs are predicted for attr', i)
                    break

                x, y = self._get_not_predicted(X, Y[i], i)

                w = self._get_schema(x, y, i)
                if w is None:
                    return False
                w = (self._simplify_schema(x, y) > 0.1).astype(np.bool, copy=False)
                w = self._logical_filter(w, i, log=log)
                self.add(w, i)
                if log:
                    self.log()
        return True

    def save(self, type_name='standard', is_reward=''):
        path = '_schemas_' + type_name + is_reward
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(self.get_schema_num()):
            np.savetxt(path + '/schema' + str(i) + '.txt', self._W[i])

    def load(self, type_name='standard', is_reward=''):
        path = '_schemas_' + type_name + is_reward
        if not os.path.exists(path):
            return
        for i in range(self.get_schema_num()):
            schema = np.loadtxt(path + '/schema' + str(i) + '.txt')
            if len(schema.shape) > 1:
                self._W[i] = schema
