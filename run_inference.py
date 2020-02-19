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
        'report_nzis_as_entities': 'all'
    }

    def __init__(self, env_type, n_episodes, n_steps):
        self.env_class = self.env_type_to_class[env_type]
        self.n_episodes = n_episodes
        self.n_steps = n_steps

    def load_schema_matrices(self):
        if not self.USE_LEARNED_SCHEMAS:
            W, R, R_weights = HardcodedSchemaVectors.gen_schema_matrices()
        else:
            dir_name = './dump'
            W = []
            R = []
            R_weights = None
            for idx in range(self.M - 1):
                file_name = 'w_{}.pkl'.format(idx)
                path = os.path.join(dir_name, file_name)
                w = np.load(path, allow_pickle=True).astype(bool)
                W.append(w)

            #for idx in range(1):
             #   file_name = '_schemas_standardreward/schema{}.txt'.format(idx)
             #   path = os.path.join(dir_name, file_name)
             #   r = np.loadtxt(path).astype(bool)
             #   R.append(r)

            _, R, _ = HardcodedSchemaVectors.gen_schema_matrices()
        return W, R, R_weights

    def loop(self):
        W, R, _ = self.load_schema_matrices()

        env = self.env_class(**self.env_params)
        env.reset()

        planner = SchemaNetwork()
        visualizer = Visualizer(None, None, None)

        for episode_idx in range(self.n_episodes):
            frame_stack = deque(maxlen=self.FRAME_STACK_SIZE)
            actions = deque()
            emergency_replanning_timer = None

            for step_idx in range(self.n_steps):
                curr_iter = episode_idx * self.n_steps + step_idx

                obs = EntityExtractor.extract(env)
                frame_stack.append(obs)

                if self.VISUALIZE_STATE:
                    # visualize env state
                    visualizer.set_iter(curr_iter)
                    visualizer.visualize_env_state(obs)

                is_planning_needed = len(actions) == 0 and emergency_replanning_timer is None \
                                     or emergency_replanning_timer == 0
                can_run_planner = len(frame_stack) == self.FRAME_STACK_SIZE
                if is_planning_needed and can_run_planner:
                    emergency_replanning_timer = None

                    planner.set_weights(W, R)
                    planner.set_curr_iter(curr_iter)

                    planned_actions = planner.plan_actions(frame_stack)
                    if planned_actions is not None:
                        actions.clear()
                        actions.extend(planned_actions)

                if actions:
                    action = actions.popleft()
                else:
                    action = 0

                    if can_run_planner:
                        if emergency_replanning_timer is None:
                            emergency_replanning_timer = self.EMERGENCY_REPLANNING_PERIOD
                        emergency_replanning_timer -= 1


                obs, reward, done, _ = env.step(action)
                if done:
                    print('END_OF_EPISODE, step_idx == {}'.format(step_idx))
                    break


def main():
    n_episodes = 1
    n_steps = 1024
    env_type = 'standard'
    assert env_type in ('standard', 'offset-paddle', 'juggling')

    runner = InferenceRunner(env_type=env_type,
                             n_episodes=n_episodes,
                             n_steps=n_steps)
    runner.loop()


if __name__ == '__main__':
    main()
