import os
from collections import deque
import numpy as np
from environment.schema_games.breakout.games import StandardBreakout, OffsetPaddleBreakout, JugglingBreakout
from model.entity_extractor import EntityExtractor
from model.inference import SchemaNetwork
from model.visualizer import Visualizer
from model.constants import Constants
from testing.testing import HardcodedSchemaVectors


class InferenceRunner(Constants):
    env_type_to_class = {
        'standard': StandardBreakout,
        'offset-paddle': OffsetPaddleBreakout,
        'juggling': JugglingBreakout
    }
    env_params = {
        'report_nzis_as_entities': 'all',
        'n_balls': 2,
    }

    def __init__(self, env_type, n_episodes, n_steps):
        self.env_class = self.env_type_to_class[env_type]
        self.n_episodes = n_episodes
        self.n_steps = n_steps

    def load_schema_matrices(self):
        dir_name = './dump'
        W = []
        R = []
        R_weights = None
        for idx in range(self.M - 1):
            file_name = 'w_{}.pkl'.format(idx)
            path = os.path.join(dir_name, file_name)
            w = np.load(path, allow_pickle=True).astype(bool)
            W.append(w)
        return W

    def _log_episode_reward(self, episode_idx, step_idx, episode_reward):
        file_path = './visualization/logs/ep_{:0{ipl}}__step_{:0{ipl}}__r_{:0{ipl}}'.format(
                        episode_idx, step_idx, episode_reward, ipl=5)

        with open(file_path, 'wt') as file:
            file.write('Amazing episode!')

    def _end_of_episode_handler(self, episode_idx, step_idx, episode_reward):
        print('END_OF_EPISODE, step_idx == {}, ep_reward == {}'.format(step_idx, episode_reward))
        self._log_episode_reward(episode_idx, step_idx, episode_reward)

    def loop(self):
        W = self.load_schema_matrices()
        _, R, _ = HardcodedSchemaVectors.gen_schema_matrices()

        env = self.env_class(**self.env_params)
        planner = SchemaNetwork()
        visualizer = Visualizer(None, None, None)

        for episode_idx in range(self.n_episodes):
            env.reset()
            reward = 0
            episode_reward = 0

            frame_stack = deque(maxlen=self.FRAME_STACK_SIZE)
            actions = deque()

            planning_timer = 0
            emergency_replanning_timer = None

            for step_idx in range(self.n_steps):
                curr_iter = episode_idx * self.n_steps + step_idx
                print('\ncurr_iter: {}'.format(curr_iter))

                obs = EntityExtractor.extract(env)
                frame_stack.append(obs)

                if self.VISUALIZE_STATE:
                    # visualize env state
                    visualizer.set_iter(curr_iter)
                    visualizer.visualize_env_state(obs)

                can_run_planner = len(frame_stack) == self.FRAME_STACK_SIZE
                #is_planning_needed = len(actions) == 0 and emergency_replanning_timer is None \
                #                     or emergency_replanning_timer == 0
                is_planning_needed = (planning_timer == 0)

                if is_planning_needed and can_run_planner:
                    emergency_replanning_timer = None

                    planner.set_weights(W, R)
                    planner.set_curr_iter(curr_iter)

                    planned_actions = planner.plan_actions(frame_stack)
                    if planned_actions is not None:
                        actions.clear()
                        actions.extend(planned_actions)

                    planning_timer = self.PLANNING_PERIOD

                if planning_timer > 0:
                    planning_timer -= 1

                if actions:
                    action = actions.popleft()
                else:
                    action = 0

                    if can_run_planner:
                        if emergency_replanning_timer is None:
                            emergency_replanning_timer = self.EMERGENCY_REPLANNING_PERIOD
                        emergency_replanning_timer -= 1


                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    print('END_OF_EPISODE, step_idx == {}'.format(step_idx))
                    break

            self._end_of_episode_handler(episode_idx, step_idx, episode_reward)


def main():
    n_episodes = 32
    n_steps = 512
    env_type = 'standard'
    assert env_type in ('standard', 'offset-paddle', 'juggling')

    runner = InferenceRunner(env_type=env_type,
                             n_episodes=n_episodes,
                             n_steps=n_steps)
    runner.loop()


if __name__ == '__main__':
    main()
