from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
import numpy as np
from model.schemanet import SchemaNet
from model.inference import SchemaNetwork
import time
from model.constants import Constants
import random


class Player(Constants):
    def __init__(self, model, reward_model, game_type=StandardBreakout):
        self.model = model
        self.reward_model = reward_model
        self.game_type = game_type
        self._memory = []
        self.rewards = []
        return

    def _transform_to_array(self, pos=0, neg=0, ent_num=94 * 117):
        return (np.zeros([ent_num, 4]) + np.array([pos, neg, 0, 0])).T

    def _uniqy(self, X):
        if len(X) == 0:
            return np.array([])
        return np.unique(X, axis=0)

    # transform data for learning:
    def _x_add_prev_time(self, action):

        X = np.vstack((matrix.transform_matrix_with_action(action=action) for matrix in self._memory[:-1]))
        X_no_actions = (X.T[:-self.ACTION_SPACE_DIM]).T
        actions = (X.T[-self.ACTION_SPACE_DIM:]).T
        X = np.concatenate((X_no_actions[:-self.N], X_no_actions[self.N:], actions[self.N:]), axis=1)
        return X

    def _y_add_prev_time(self):
        return np.vstack((matrix.matrix.T for matrix in self._memory[2:]))

    def _check_for_update(self, X, old_state):
        old_state = np.array(old_state)
        update = []
        for entity in X:
            tmp = (entity == old_state)
            if type(tmp) == bool:
                if not tmp:
                    update.append(entity)
            elif type(tmp.all(axis=1)) == bool:
                if not tmp.all(axis=1):
                    update.append(entity)
            elif not tmp.all(axis=1).any():
                update.append(entity)
        update = self._uniqy(update)
        return len(update), np.array(update)

    def _get_action_for_reward(self, env):
        pos_ball = 0
        pos_paddle = 0
        for ball in env.balls:
            if ball.is_entity:
                for state, eid in env.parse_object_into_pixels(ball):
                    pos_ball = list(state.keys())[0][1]

        if env.paddle.is_entity:
            for state, eid in env.parse_object_into_pixels(env.paddle):
                pos_paddle = list(state.keys())[0][1]

        if pos_ball[1] < pos_paddle[1]:
            return 1
        return 2

    def _free_mem(self):
        self._memory = []

    def play(self, game_type=StandardBreakout,
             step_num=3,
             learning_freq=3,
             log=False, cheat=False):
        old_state = []

        flag = 0
        y = np.array([0, 0, 0, 0, 0])


        for i in range(step_num):
            env = game_type(return_state_as_image=False)
            # done = False
            env.reset()
            if cheat:
                self.model.add_cheat_schema()

            j = 0
            action = 0
            state, reward, done, _ = env.step(action)

            while not done:

                self._memory.append(FeatureMatrix(env))

                # learn new schemas
                if j % learning_freq == 2:

                    # transform data for learning
                    X = self._x_add_prev_time(action)
                    y = self._y_add_prev_time()

                    # get new unique windows to learn reward
                    ent_num, update = self._check_for_update(X, old_state)

                    # learn reward
                    if len(update) != 0:
                        if log:
                            print('learning reward', reward > 0, reward)
                        y_r = self._transform_to_array(reward > 0, reward < 0, ent_num=ent_num)
                        old_state += list(update)
                        self.reward_model.fit(update, y_r)

                    # learn env state:

                    self.model.fit(X, y)

                    # make a decision
                    r = random.randint(1, 3)
                    if flag == 0 and r == 1:
                        action = self._get_action_for_reward(env)
                    else:
                        print('$$$$$$', flag)
                        start = time.time()

                        W = [w == 1 for w in self.model._W]
                        R = [self.reward_model._W[0] == 1, self.reward_model._W[1] == 1]

                        if all(w.shape[1] > 1 for w in W):
                            decision_model = SchemaNetwork(W, R, self._memory[-self.FRAME_STACK_SIZE:])
                            decision_model.set_curr_iter(j)
                            action = decision_model.plan_actions()[0] + 1
                        else:
                            action = self._get_action_for_reward(env)

                        end = time.time()
                        print("--- %s seconds ---" % (end - start))
                    self._free_mem()

                j += 1
                print('action:', action)
                state, reward, done, _ = env.step(action)
                if reward == 1:
                    if flag == 0:
                        print('PLAYER CHANGED')
                    flag = 1

                self.rewards.append(reward)

            if log:
                print('step:', i)

        self.model.save()
