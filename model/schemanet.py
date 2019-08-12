import numpy as np
from scipy.optimize import linprog


class SchemaNet:
    def __init__(self, N=0, M=53, A=2, L=100, window_size=2):
        self._M = M
        self.neighbour_num = (window_size * 2 + 1) ** 2 + A
        self._W = [np.zeros(self.neighbour_num * M) + 1] * M
        self.solved = np.array([])
        self._A = A
        self._L = L

    def log(self):
        print('current net:\n', self._W)

    def predict_attr(self, X, i):
        if len(self._W[i].shape) == 1:
            return (X == 0) @ self._W[i] == 0
        return ((X == 0) @ self._W[i] == 0).any(axis=1) != 0

    def predict(self, X):
        return [self.predict_attr(X, i) for i in range(self._M)]

    def add(self, schemas, i):
        self._W[i] = np.vstack((self._W[i].T, schemas.T)).T

    def solve_lp(self, zero_pred, c, A_ub, b_ub, A_eq, b_eq):
        if len(zero_pred) == 0:
            return linprog(c=c, A_eq=A_eq, b_eq=b_eq).x.round(2)
        else:
            return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq).x.round(2)

    def get_not_predicted(self, X, y, i):
        ind = (self.predict_attr(X, i) != y) | (y == 0)
        return X[ind], y[ind]

    def get_next_to_predict(self, X, y, i):
        ind = (self.predict_attr(X, i) != y) * (y == 1)
        return X[ind][0]

    def get_schema(self, X, y, i):

        zero_pred = X[y == 0]
        ones_pred = X[y == 1]

        self.solved = np.array([self.get_next_to_predict(X, y, i)])

        c = (1 - ones_pred).sum(axis=0)
        A_eq = 1 - self.solved
        b_eq = np.zeros(self.solved.shape[0])
        A_ub = zero_pred - 1
        b_ub = np.zeros(zero_pred.shape[0]) - 1

        w = self.solve_lp(zero_pred, c, A_ub, b_ub, A_eq, b_eq)

        preds = ((X == 0) @ w) == 0
        self.solved = np.vstack((self.solved, X[preds * (self.predict_attr(X, i) == 0)]))
        return w

    def simplify_schema(self, X, y):

        zero_pred = X[y == 0]

        c = np.zeros(self.neighbour_num * self._M) + 1
        A_eq = (1 - self.solved)
        b_eq = np.zeros(self.solved.shape[0])
        A_ub = (zero_pred - 1)
        b_ub = np.zeros(zero_pred.shape[0]) - 1

        return self.solve_lp(zero_pred, c, A_ub, b_ub, A_eq, b_eq)

    def fit(self, X, Y, log=True):

        for i in range(self._M):

            for j in range(self._L):

                if (self.predict_attr(X, i) == Y[i]).all():
                    if log:
                        print('all attrs are predicted for attr', i)
                    break

                x, y = self.get_not_predicted(X, Y[i], i)

                self.get_schema(x, y, i)
                w = self.simplify_schema(x, y)
                self.add(w, i)
                if log:
                    self.log()


if __name__ == '__main__':
    X = np.array([[0, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1]])
    y = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 1, 0, 1, 1]])
    schemanet = SchemaNet(M=5, A=0, window_size=0)
    schemanet.fit(X, y)


