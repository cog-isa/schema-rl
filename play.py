from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
import numpy as np
from model.schemanet import SchemaNet


def play(model,
         game_type=StandardBreakout,
         step_num=3,
         window_size=20,
         attrs_num=4,
         action_space=2,
         attr_num=94*117,
         learning_freq=1):
    memory = []
    reward_mem = []

    reward_model = SchemaNet(M=attrs_num*attr_num, A=2, L=100, window_size=0)

    for i in range(step_num):
        env = game_type(return_state_as_image=False)
        done = False
        env.reset()
        while not done:
            matrix = FeatureMatrix(env, attrs_num=attrs_num, window_size=window_size, action_space=action_space)
            memory.append(matrix)
            # make a decision
            action = np.random.randint(2) + 1

            state, reward, done, _ = env.step(action)
            reward_mem.append(reward)
            # TODO: transform_matrix takes terribly long
            if i % learning_freq == 0:
                X = np.vstack((matrix.transform_matrix(action=action) for matrix in memory))
                y = np.vstack((matrix.matrix.T for matrix in memory))
                model.fit(X, y)
                memory = []
            if i % 10 == 9:
                X = np.array([np.vstack(matrix.matrix) for matrix in memory])
                y = np.array(reward_mem)
                reward_model.fit(X, y)

            print(reward, end='; ')
        print('step:', i)
        return reward_model


if __name__ == '__main__':
    window_size = 2
    model = SchemaNet(M=4, A=2, window_size=window_size)
    reward_model = play(model, step_num=20, window_size=window_size)
    f = open('matrix.txt', 'w')
    f.write(model._W)
    f.write('*'*50 + '\n')
    f.write(reward_model._W)
    f.close()
