from collections import deque
import time
import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
from model.constants import Constants
from model.visualizer import Visualizer
from model.entity_extractor import EntityExtractor
from model.schema_learner import GreedySchemaLearner
from model.shaper import Shaper


class DataLoader(Constants):
    def __init__(self, shaper):
        self._shaper = shaper

    def make_batch(self, observations, actions):
        frame_stack = [observations[idx] for idx in range(self.FRAME_STACK_SIZE)]
        action = actions[self.FRAME_STACK_SIZE - 1]
        augmented_entities = self._shaper.transform_matrix(frame_stack, action=action)
        target = observations[self.FRAME_STACK_SIZE][:, :self.N_PREDICTABLE_ATTRIBUTES]
        batch = GreedySchemaLearner.Batch(augmented_entities, target)
        return batch


class LearningRunner(Constants):
    env_params = {
        'report_nzis_as_entities': 'all',
        'num_lives': 10 ** 9 + 7
    }

    def __init__(self, n_episodes, n_steps):
        self.n_episodes = n_episodes
        self.n_steps = n_steps

        self._paddle_keypoint_idx = None
        self._keypoint_timer = None
        self.KEYPOINT_TIMER = 30

    def _get_hardcoded_action(self, env, eps=0.0):
        if np.random.uniform() < eps:
            chosen_action = np.random.choice(self.ACTION_SPACE_DIM)
        else:
            ball_x = EntityExtractor.get_ball_x(env)
            targets = EntityExtractor.get_paddle_keypoints(env)

            if not self._keypoint_timer:
                self._paddle_keypoint_idx = np.random.choice(len(targets))
                self._keypoint_timer = self.KEYPOINT_TIMER
            else:
                self._keypoint_timer -= 1

            target = targets[self._paddle_keypoint_idx]
            if ball_x < target:
                chosen_action = 1
            elif ball_x > target:
                chosen_action = 2
            else:
                chosen_action = 0

        return chosen_action

    def learn(self):
        env = StandardBreakout(**self.env_params)
        shaper = Shaper()
        data_loader = DataLoader(shaper)
        learner = GreedySchemaLearner()
        visualizer = Visualizer(None, None, None)

        env.reset()

        for episode_idx in range(self.n_episodes):
            observations = deque(maxlen=self.LEARNING_BATCH_SIZE)
            actions = deque(maxlen=self.LEARNING_BATCH_SIZE)

            for step_idx in range(self.n_steps):
                curr_iter = episode_idx * self.n_steps + step_idx
                print('\ncurr_iter: {}'.format(curr_iter))

                obs = EntityExtractor.extract(env)
                observations.append(obs)

                # hardcoded action choice
                chosen_action = self._get_hardcoded_action(env)
                actions.append(chosen_action)

                # visualizing
                if self.VISUALIZE_STATE:
                    visualizer.set_iter(curr_iter)
                    visualizer.visualize_env_state(obs)

                # learning
                if len(observations) >= self.LEARNING_BATCH_SIZE:
                    batch = data_loader.make_batch(observations, actions)

                    learner.set_curr_iter(curr_iter)
                    learner.take_batch(batch)

                    is_flush_needed = curr_iter == self.n_episodes * self.n_steps - 1
                    if curr_iter % self.LEARNING_PERIOD == 0 or is_flush_needed:
                        learner.learn()

                obs, reward, done, _ = env.step(chosen_action)
                if done:
                    print('END_OF_EPISODE, step_idx == {}'.format(step_idx))
                    break


def main():
    n_episodes = 1
    n_steps = 2048

    start_time = time.time()
    runner = LearningRunner(n_episodes=n_episodes,
                            n_steps=n_steps)
    runner.learn()
    print('Elapsed time: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()