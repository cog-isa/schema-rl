from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
import numpy as np
from model.schemanet import SchemaNet
from model.inference import SchemaNetwork


def transform_to_array(pos=0, neg=0, ent_num=94*117):
    return (np.zeros([ent_num, 4]) + np.array([pos, neg, 0, 0])).T


def check_for_update(X, old_state):
    old_state = np.array(old_state)
    update = []
    for entity in X:
        if entity not in old_state:
            update.append(entity)
    return len(update), np.array(update)


def play(model, reward_model,
         game_type=StandardBreakout,
         step_num=3,
         window_size=20,
         attrs_num=4,
         action_space=2,
         attr_num=94*117,
         learning_freq=2):
    memory = []
    reward_mem = []
    old_state  = []

    for i in range(step_num):
        env = game_type(return_state_as_image=False)
        done = False
        env.reset()
        j = 0
        while not done:

            matrix = FeatureMatrix(env, attrs_num=attrs_num, window_size=window_size, action_space=action_space)
            memory.append(matrix)
            # make a decision
            decision_model = SchemaNetwork([w==1 for w in model._W],
                                           [reward_model._W[0] ==1, reward_model._W[1] ==1],
                                           matrix)

            actions = decision_model.plan_actions()
            print(actions)
            action = np.random.randint(2) + 1

            state, reward, done, _ = env.step(action)
            reward_mem.append(reward)
            # TODO: transform_matrix takes terribly long
            if j % learning_freq == 1:
                X = np.vstack((matrix.transform_matrix_with_action(action=action) for matrix in memory[:-1]))
                y = np.vstack((matrix.matrix.T for matrix in memory[1:]))

                ent_num, update = check_for_update(X, old_state)
                if len(update) != 0:
                    y_r = transform_to_array(reward > 0, reward < 0, ent_num=ent_num)
                    old_state += list(update)
                    reward_model.fit(update, y_r)
                reward_mem = []

                model.fit(X, y)
                memory = []
            j += 1

            print('     ', reward, end='; ')
        print('step:', i)


if __name__ == '__main__':
    window_size = 2
    model = SchemaNet(M=4, A=2, window_size=window_size)
    reward_model = SchemaNet(M=4, A=2, window_size=window_size)
    play(model, reward_model, step_num=2, window_size=window_size)
    for i in range(len(model._W)):
        np.savetxt('matrix'+str(i)+'.txt', model._W[i])
    for i in range(len(reward_model._W)):
        np.savetxt('matrix_reward'+str(i)+'.txt', reward_model._W[i])
