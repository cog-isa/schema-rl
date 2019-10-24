import os
from collections import deque
import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
from model.inference import SchemaNetwork
from model.constants import Constants
from testing.testing import HardcodedSchemaVectors


class Runner(Constants):
    def __init__(self, n_episodes, n_steps):
        self.n_episodes = n_episodes
        self.n_steps = n_steps

    def load_schema_matrices(self, generate=True):
        if generate:
            W = HardcodedSchemaVectors.gen_attribute_schema_matrices()
            for w in W:
                print(w.shape)
            R = W.copy()  # temporarily
        else:
            dir_name = './dump'
            W = []
            R = []
            for idx in range(self.M):
                file_name = 'w_{}'.format(idx)
                path = os.path.join(dir_name, file_name)
                w = np.load(path, allow_pickle=True)
                W.append(w)
            for idx in range(2):
                file_name = 'r_{}'.format(idx)
                path = os.path.join(dir_name, file_name)
                r = np.load(path, allow_pickle=True)
                R.append(r)
        return W, R

    def loop(self):
        W, R = self.load_schema_matrices()

        env = StandardBreakout()
        env.reset()

        for episode_idx in range(self.n_episodes):
            frame_stack = deque(maxlen=self.FRAME_STACK_SIZE)

            for step_idx in range(self.n_steps):
                obs = FeatureMatrix(env).matrix
                frame_stack.append(obs)

                model = SchemaNetwork(W, R, frame_stack)
                model.set_curr_iter(episode_idx * self.n_steps + step_idx)
                actions = model.plan_actions()
                action = actions[0]

                obs, reward, done, _ = env.step(action)
                if done:
                    print('END_OF_EPISODE, step_idx == {}'.format(step_idx))
                    break


def main():
    n_episodes = 1
    n_steps = 8

    runner = Runner(n_episodes=n_episodes,
                    n_steps=n_steps)
    runner.loop()


if __name__ == '__main__':
    main()
