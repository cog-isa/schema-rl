from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
import numpy as np
from model.schemanet import SchemaNet
from model.inference import SchemaNetwork
import time
import model.visualisation as vis


def transform_to_array(pos=0, neg=0, ent_num=94*117):
    return (np.zeros([ent_num, 4]) + np.array([pos, neg, 0, 0])).T


def uniqy(X):
    if len(X) == 0:
        return np.array([])
    return np.unique(X, axis=0)


# transform data for learning:
def x_add_prev_time(memory, action, action_space=2, attr_num=94 * 117, log=False):

    X = np.vstack((matrix.transform_matrix_with_action(action=action) for matrix in memory[:-1]))
    if log:
        X = np.vstack((matrix.transform_matrix_with_action(action=action) for matrix in memory))
    print('shape', X.shape)
    X = np.concatenate(((X.T[:-action_space]).T[attr_num:], X[:-attr_num]), axis=1)
    print('shape', X.shape)
    return X


def y_add_prev_time(memory, log=True):
    if log:
        print('y_add_prev_time', len(memory[2:]), memory[-1].matrix.T.shape)
    return np.vstack((matrix.matrix.T for matrix in memory[2:]))


def check_for_update(X, old_state):
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
    update = uniqy(update)
    return len(update), np.array(update)


def get_action_for_reward(env):
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


def play(model, reward_model,
         game_type=StandardBreakout,
         step_num=3,
         window_size=2,
         attrs_num=4,
         action_space=2,
         attr_num=94*117,
         learning_freq=3,
         log=False,
         num_priv_steps=2):
    memory = []
    reward_mem = []
    old_state  = []

    flag = 1
    y = np.array([0, 0, 0, 0, 0])

    for i in range(step_num):
        env = game_type(return_state_as_image=False)

        done = False
        env.reset()
        j = 0
        while not done:

            matrix = FeatureMatrix(env, attrs_num=attrs_num, window_size=window_size, action_space=action_space)
            memory.append(matrix)

            # make a decision
            if flag == 0:
                action = get_action_for_reward(env)
            else:
                start = time.time()

                W = [w == 1 for w in model._W]
                R = [reward_model._W[0] == 1, reward_model._W[1] == 1]

                if all(w.shape[1] > 1 for w in W):
                    decision_model = SchemaNetwork(W, R, memory[-num_priv_steps:])
                    decision_model.set_curr_iter(j)
                    action = decision_model.plan_actions()[0] + 1
                else:
                    action = get_action_for_reward(env)

                end = time.time()
                print("--- %s seconds ---" % (end - start))

            print('action:', action)

            state, reward, done, _ = env.step(action)
            if reward == 1:
                if flag == 0:
                    print('PLAYER CHANGED')
                flag = 1

            reward_mem.append(reward)

            # learn new schemas
            if j % learning_freq == 2:

                # transform data for learning
                X = x_add_prev_time(memory, action)

                y_prev = y.T[0]
                y = y_add_prev_time(memory)

                # get new unique windows to learn reward
                ent_num, update = check_for_update(X, old_state)

                # learn reward
                if len(update) != 0:
                    if log:
                        print('learning reward', reward > 0, reward)
                    y_r = transform_to_array(reward > 0, reward < 0, ent_num=ent_num)
                    old_state += list(update)
                    reward_model.fit(update, y_r)

                reward_mem = []

                # learn env state:
                model.fit(X, y)

                if log:
                    # predict for T steps:
                    T = 8
                    action = 1
                    feature_matrix = FeatureMatrix(env, attrs_num=attrs_num, window_size=window_size,
                                                   action_space=action_space)
                    stats = []
                    tmp_mem = memory[-(num_priv_steps):]

                    model.visualize_schemas()
                    for i in range(T):
                        print("@@@", i)
                        X = x_add_prev_time(tmp_mem, action, log=True)
                        y = model.predict(X).T

                        stats.append(y)
                        feature_matrix.matrix = y
                        tmp_mem[-2].matrix = tmp_mem[-1].matrix[:]
                        tmp_mem[-1].matrix = y[:]
                        #'''

                    img = vis.img_average(stats)
                    vis.save_img(img, img_name='images/img' + str(j) + '.png', log=False)

                memory = []
            j += 1

            if log:
                print('     ', reward, end='; ')
        if log:
            print('step:', i)


if __name__ == '__main__':
    window_size = 2
    model = SchemaNet(M=4, A=2, window_size=window_size)
    reward_model = SchemaNet(M=4, A=2, window_size=window_size)
    play(model, reward_model, step_num=2, window_size=window_size, log=True)
    for i in range(len(model._W)):
        np.savetxt('matrix'+str(i)+'.txt', model._W[i])
    for i in range(len(reward_model._W)):
        np.savetxt('matrix_reward'+str(i)+'.txt', reward_model._W[i])
