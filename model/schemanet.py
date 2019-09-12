import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm
import time
from termcolor import colored


class SchemaNet:
    def __init__(self, N=0, M=53, A=2, L=100, window_size=2):
        self._M = M
        self.neighbour_num = (window_size * 2 + 1) ** 2
        print('neighbour_num', self.neighbour_num)
        self._W = [np.zeros([self.neighbour_num * M + A, 1]) + 1] * M
        self.solved = np.array([])
        self._A = A
        self.window_size = window_size
        self._L = L
        self._R = [np.zeros(self.neighbour_num * M + A) + 1] * 2
        self.reward = []
        self.memory = []

    def log(self):
        print('current net:\n', [self._W[i].shape for i in range(len(self._W))])

    def print(self):
        print('current net:\n')
        for i in range(len(self._W)):
            print('   ' * i, self._W[i].T)

    def predict_attr(self, X, i):
        if len(self._W[i].shape) == 1:
            return (X == 0) @ self._W[i] == 0
        return ((X == 0) @ self._W[i] == 0).any(axis=1) != 0

    def schema_predict_attr(self, X, i):
        return ((X == 0) @ self._W[i] == 0) != 0

    def predict_reward(self, X, is_pos=1):
        if len(self._R[is_pos].shape) == 1:
            return (X == 0) @ self._R[is_pos] == 0
        return ((X == 0) @ self._R[is_pos] == 0).any(axis=1) != 0

    def predict(self, X):
        return np.array([self.predict_attr(X, i) for i in range(self._M)])

    def add(self, schemas, i):
        self._W[i] = np.vstack((self._W[i].T, schemas.T)).T

    def scipy_solve_lp(self, zero_pred, c, A_ub, b_ub, A_eq, b_eq, options={'maxiter': 2, "disp": False}):
        if len(zero_pred) == 0:
            return linprog(c=c, A_eq=A_eq, b_eq=b_eq, options=options).x.round(2)
        else:
            return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, options=options).x.round(2)

    def get_not_predicted(self, X, y, i):
        ind = (self.predict_attr(X, i) != y) | (y == 0)
        return X[ind], y[ind]

    def reverse_transform_print(self, y):
        for i in range(self.window_size * 2 + 1):
            for j in range(self.window_size * 2 + 1):
                ind = i * (self.window_size * 2 + 1) * 4 + j * 4
                if y[ind] == 1:
                    print(colored('*', 'red'), end='')
                elif y[ind + 1] == 1:
                    print(colored('*', 'cyan'), end='')
                elif y[ind + 2] == 1:
                    print(colored('*', 'green'), end='')
                elif y[ind + 3] == 1:
                    print(colored('*', 'grey'), end='')
                else:
                    print(colored('*', 'white'), end='')
            print()

    def visualize_schemas(self):
        for W in self._W:
            for schema in W.T:
                self.reverse_transform_print(schema)
                print('-' * 50)

    def get_next_to_predict(self, X, y, i, log=False):
        ind = (self.predict_attr(X, i) != y) * (y == 1)

        if ind.sum() == 0:
            print(colored('FATAL ERROR', 'red'))
            return None
        result = X[ind][0]
        return result

    def get_schema(self, X, y, i, log=True):

        zero_pred = X[y == 0]
        ones_pred = X[y == 1]

        new_ent = self.get_next_to_predict(X, y, i, log=log)
        self.solved = np.array([new_ent])

        c = (1 - ones_pred).sum(axis=0)
        A_eq = 1 - self.solved
        b_eq = np.zeros(self.solved.shape[0])
        A_ub = zero_pred - 1
        b_ub = np.zeros(zero_pred.shape[0]) - 1
        w = self.scipy_solve_lp(zero_pred, c, A_ub, b_ub, A_eq, b_eq)

        preds = ((X == 0) @ w) == 0
        self.solved = np.vstack((self.solved, X[preds * (self.predict_attr(X, i) == 0)]))
        return w

    def simplify_schema(self, X, y):

        zero_pred = X[y == 0]

        c = np.zeros(self.neighbour_num * self._M + self._A)
        A_eq = (1 - self.solved)
        b_eq = np.zeros(self.solved.shape[0])
        A_ub = (zero_pred - 1)
        b_ub = np.zeros(zero_pred.shape[0]) - 1

        return self.scipy_solve_lp(zero_pred, c, A_ub, b_ub, A_eq, b_eq)

    def actuality_check_attr(self, X, y, i):
        if len(self._W[i].shape) == 1:
            return np.zeros(self._W[i].shape[0])
        pred = self.schema_predict_attr(X, i).T
        return ((y[i] - pred) == -1)

    def remove_wrong_schemas(self, X, y):
        for i in range(self._M):
            if len(self._W[i].shape) == 1:
                break
            wrong_ind = self.actuality_check_attr(X, y, i).sum(axis=1)

            if (wrong_ind.sum()) != 0:
                print('outdated schema was detected for attribute', i)

            self._W[i] = (self._W[i].T[wrong_ind == 0]).T

    def fit(self, X, Y, log=True):
        tmp, ind = np.unique(X, return_index=True, axis=0)
        X = X[ind]
        Y = (Y.T[ind]).T

        self.remove_wrong_schemas(X, Y)

        for i in (range(self._M)):

            for j in (range(self._L)):

                if isinstance((self.predict_attr(X, i) == Y[i]), np.ndarray):
                    if (self.predict_attr(X, i) == Y[i]).all():
                        if log:
                            print('all attrs are predicted for attr', i)
                        break
                else:
                    if self.predict_attr(X, i) == Y[i]:
                        if log:
                            print('all attrs are predicted for attr', i)
                        break

                x, y = self.get_not_predicted(X, Y[i], i)

                self.get_schema(x, y, i)
                w = (self.simplify_schema(x, y) > 0.1) * 1
                self.add(w, i)
                if log:
                    self.log()


if __name__ == '__main__':
    X = np.array([[0, 1, 0, 1, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0]])
    y = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 1, 1]])
    schemanet = SchemaNet(M=6, A=0, window_size=0)
    print('pred', schemanet.schema_predict_attr(X, 1), schemanet.actuality_check_attr(X, y, 1))
    schemanet.fit(X, y)
    schemanet.print()
    print(schemanet.predict_attr(X, 5))
    #print(torch.cuda.is_available())