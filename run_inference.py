from collections import deque
import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
from model.inference import SchemaNetwork
from model.constants import Constants


def load_schema_matrices():
    W = []
    R = []
    for idx in range(self.M):
        w = np.load(f'w_{idx}')
        W.append(w)
    for idx in range(2):
        r = np.load(f'r_{idx}')
        R.append(r)
    return W, R

def main():
    W, R = load_schema_matrices()

    env = StandardBreakout()
    env.reset()

    frame_stack = deque(maxlen=Constants.FRAME_STACK_SIZE)
    obs = FeatureMatrix(env,
                        shape=(Constants.SCREEN_HEIGHT, Constants.SCREEN_WIDTH),
                        attrs_num=Constants.M,
                        window_size=Constants.NEIGHBORHOOD_RADIUS,
                        action_space=Constants.ACTION_SPACE_DIM).matrix

    frame_stack.append(obs)

    model = SchemaNetwork(W, R, frame_stack)

