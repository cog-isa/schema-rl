from collections import deque
import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
from model.inference import SchemaNetwork
from model.constants import Constants

class Runner(Constants):
    def __init__(self, n_episodes, n_steps):
        self.n_episodes = n_episodes
        self.n_steps = n_steps

    def load_schema_matrices(self):
        W = []
        R = []
        for idx in range(self.M):
            w = np.load('w_{}'.format(idx))
            W.append(w)
        for idx in range(2):
            r = np.load('r_{}'.format(idx))
            R.append(r)
        return W, R

    def loop(self):
        W, R = self.load_schema_matrices()

        env = StandardBreakout()
        env.reset()

        for episode_idx in range(self.n_episodes):
            frame_stack = deque(maxlen=self.FRAME_STACK_SIZE)

            for step_idx in range(self.n_steps):
                obs = FeatureMatrix(env,
                                    shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH),
                                    attrs_num=self.M,
                                    window_size=self.NEIGHBORHOOD_RADIUS,
                                    action_space=self.ACTION_SPACE_DIM).matrix
                frame_stack.append(obs)

                model = SchemaNetwork(W, R, frame_stack)
                model.set_curr_iter(episode_idx * self.n_steps + step_idx)
                action = model.plan_actions()

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
