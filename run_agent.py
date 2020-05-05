import os
from collections import deque
import time
import itertools

import numpy as np

from environment.schema_games.breakout.games import StandardBreakout, OffsetPaddleBreakout, JugglingBreakout
from model.entity_extractor import EntityExtractor
from model.inference import SchemaNetwork
from model.visualizer import Visualizer
from model.constants import Constants as C
from model.schema_learner import GreedySchemaLearner
from model.shaper import Shaper
from testing.testing import HardcodedDeltaSchemaVectors


class LearningHandler:
    """
    Instantiate every episode
    """
    def __init__(self, learner, shaper, last_iter):
        self._learner = learner
        self._shaper = shaper
        self._last_iter = last_iter

        self._observations = deque(maxlen=C.LEARNING_BATCH_SIZE)
        self._actions_taken = deque(maxlen=C.LEARNING_BATCH_SIZE)

    def _make_batch(self, reward):
        frame_stack = [self._observations[idx] for idx in range(C.FRAME_STACK_SIZE)]
        action = self._actions_taken[C.FRAME_STACK_SIZE - 1]
        augmented_entities = self._shaper.transform_matrix(frame_stack, action=action)

        next_state = self._observations[C.FRAME_STACK_SIZE][:, :C.N_PREDICTABLE_ATTRIBUTES]
        last_state = self._observations[C.FRAME_STACK_SIZE - 1][:, :C.N_PREDICTABLE_ATTRIBUTES]
        target_creation = np.subtract(next_state, last_state, dtype=int).clip(min=0).astype(bool)
        target_destruction = np.subtract(last_state, next_state, dtype=int).clip(min=0).astype(bool)

        rewards = np.full(target_creation.shape[0], reward) == 1
        batch = GreedySchemaLearner.Batch(augmented_entities, target_creation, target_destruction, rewards)
        return batch

    def learn(self, obs, chosen_action, reward, curr_iter):
        if not (C.DO_LEARN_ATTRIBUTE_PARAMS or C.DO_LEARN_REWARD_PARAMS):
            return

        self._observations.append(obs)
        self._actions_taken.append(chosen_action)

        if len(self._observations) >= C.LEARNING_BATCH_SIZE:
            print('adding batch to learner')
            batch = self._make_batch(reward)

            self._learner.set_curr_iter(curr_iter)
            self._learner.take_batch(batch)

            is_flush_needed = (curr_iter == self._last_iter)
            if curr_iter % C.LEARNING_PERIOD == 0 or is_flush_needed:
                print('Launching learning procedure...')
                self._learner.learn()


class PlanningHandler:
    """
    Instantiate every episode
    """
    def __init__(self, planner, env):
        self._planner = planner
        self._env = env
        self._frame_stack = deque(maxlen=C.FRAME_STACK_SIZE)
        self._planned_actions = deque()

        self._planning_timer = 0
        self._emergency_planning_timer = None

        # stuff for hardcoded action
        self._paddle_keypoint_idx = None
        self._keypoint_timer = None
        self._KEYPOINT_TIMER = 30

    def _get_hardcoded_action(self, eps=0.0):
        if np.random.uniform() < eps:
            chosen_action = np.random.choice(C.ACTION_SPACE_DIM)
        else:
            ball_x = EntityExtractor.get_ball_x(self._env)
            targets = EntityExtractor.get_paddle_keypoints(self._env)

            if not self._keypoint_timer:
                self._paddle_keypoint_idx = np.random.choice(len(targets))
                self._keypoint_timer = self._KEYPOINT_TIMER
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

    def plan(self, obs, W_pos, W_neg, R, curr_iter, reward):
        if C.PLANNING_TYPE == 'hardcoded':
            return self._get_hardcoded_action()
        elif C.PLANNING_TYPE == 'random':
            return np.random.choice(C.ACTION_SPACE_DIM)
        elif C.PLANNING_TYPE == 'agent':
            pass
        else:
            assert False

        self._frame_stack.append(obs)

        are_weights_ok = all(matrix.size for matrix in itertools.chain(W_pos, W_neg, R))
        can_run_planner = are_weights_ok and len(self._frame_stack) == C.FRAME_STACK_SIZE

        if C.USE_EMERGENCY_PLANNING:
            is_planning_needed = len(self._planned_actions) == 0 and self._emergency_planning_timer is None \
                                 or self._emergency_planning_timer == 0
        else:
            is_planning_needed = (self._planning_timer == 0)

        if is_planning_needed and can_run_planner:
            print('Launching planning...')

            # handle timers
            self._emergency_planning_timer = None
            self._planning_timer = C.PLANNING_PERIOD

            # run planning
            self._planner.set_weights(W_pos, W_neg, R)
            self._planner.set_curr_iter(curr_iter)
            actions = self._planner.plan_actions(self._frame_stack)
            if actions is not None:
                self._planned_actions.clear()
                self._planned_actions.extend(actions)

        # choose from plan next action to take
        if self._planned_actions:
            chosen_action = self._planned_actions.popleft()

            # if this was last planned action, pause planning for a while
            if len(self._planned_actions) == 0:
                self._emergency_planning_timer = C.EMERGENCY_PLANNING_PERIOD
        else:
            chosen_action = np.random.choice(C.ACTION_SPACE_DIM)

            if can_run_planner:
                if self._emergency_planning_timer is None:
                    self._emergency_planning_timer = C.EMERGENCY_PLANNING_PERIOD
                self._emergency_planning_timer -= 1

        if self._planning_timer > 0:
            self._planning_timer -= 1

        return chosen_action


class Runner:
    def __init__(self, env_type, env_params, n_episodes, n_steps):
        self._env_class = {'standard': StandardBreakout,
                          'offset-paddle': OffsetPaddleBreakout,
                          'juggling': JugglingBreakout
                           }[env_type]

        self._env_params = env_params
        self._env_params.update({'report_nzis_as_entities': 'all'})

        self._n_episodes = n_episodes
        self._n_steps = n_steps

        self.hc_W_pos, self.hc_W_neg, self.hc_R = HardcodedDeltaSchemaVectors.gen_schema_matrices()


    @staticmethod
    def _load_dumped_params():
        dir_name = './dump'

        W_pos, W_neg, R = [], [], []
        names = ['w_pos', 'w_neg', 'r']
        matrix_counts = [C.N_PREDICTABLE_ATTRIBUTES, C.N_PREDICTABLE_ATTRIBUTES, 1]

        for params, name, n_matrices in zip((W_pos, W_neg, R), names, matrix_counts):
            for idx in range(n_matrices):
                file_name = name + '_{}.npy'.format(idx)
                path = os.path.join(dir_name, file_name)
                matrix = np.load(path, allow_pickle=False)
                assert matrix.dtype == bool
                params.append(matrix)

        return W_pos, W_neg, R

    def _log_episode_reward(self, episode_idx, step_idx, episode_reward):
        file_path = './visualization/logs/ep_{:0{ipl}}__step_{:0{ipl}}__r_{:0{ipl}}'.format(
                        episode_idx, step_idx, episode_reward, ipl=5)

        with open(file_path, 'wt') as file:
            file.write('Amazing episode!')

    def _end_of_episode_handler(self, episode_idx, step_idx, episode_reward):
        print('END_OF_EPISODE, step_idx == {}, ep_reward == {}'.format(step_idx, episode_reward))
        self._log_episode_reward(episode_idx, step_idx, episode_reward)

    def loop(self):
        env = self._env_class(**self._env_params)
        shaper = Shaper()
        visualizer = Visualizer(None, None, None)

        learner = GreedySchemaLearner()
        if C.DO_PRELOAD_DUMP_PARAMS:
            W_pos, W_neg, R = self._load_dumped_params()
            learner.set_params(W_pos, W_neg, R)

        planner = SchemaNetwork()

        last_iter = self._n_episodes * self._n_steps - 1

        for episode_idx in range(self._n_episodes):
            env.reset()
            reward = 0
            episode_reward = 0
            step_idx = 0

            learning_handler = LearningHandler(learner, shaper, last_iter)
            planning_handler = PlanningHandler(planner, env)

            for step_idx in range(self._n_steps):
                curr_iter = episode_idx * self._n_steps + step_idx
                print('\ncurr_iter: {}'.format(curr_iter))

                obs = EntityExtractor.extract(env)
                if C.VISUALIZE_STATE:
                    visualizer.set_iter(curr_iter)
                    visualizer.visualize_env_state(obs)

                # --- planning ---
                W_pos, W_neg, R = learner.get_params()

                if C.DO_PRELOAD_HANDCRAFTED_ATTRIBUTE_PARAMS:
                    W_pos = self.hc_W_pos
                    W_neg = self.hc_W_neg
                if C.DO_PRELOAD_HANDCRAFTED_REWARD_PARAMS:
                    R = self.hc_R

                for params in (W_pos, W_neg, R):
                    for idx, matrix in enumerate(params):
                        if not matrix.size:
                            params[idx] = np.ones((C.SCHEMA_VEC_SIZE, 1), dtype=bool)

                chosen_action = planning_handler.plan(obs, W_pos, W_neg, R, curr_iter, reward)

                # --- learning ---
                learning_handler.learn(obs, chosen_action, reward, curr_iter)

                obs, reward, done, _ = env.step(chosen_action)
                episode_reward += reward
                if done:
                    break

            self._end_of_episode_handler(episode_idx, step_idx, episode_reward)


def main():
    n_episodes = 7
    n_steps = 5000

    env_type = 'standard'  # ('standard', 'offset-paddle', 'juggling')
    env_params = {
        'num_lives': 10 ** 9 + 7,
        'n_balls': 1
    }

    start_time = time.time()
    runner = Runner(env_type=env_type,
                    env_params=env_params,
                    n_episodes=n_episodes,
                    n_steps=n_steps)
    runner.loop()
    print('Elapsed time: {}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
