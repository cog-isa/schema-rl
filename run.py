import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
from model.inference import SchemaNetwork


W = [np.full((162, 5), True) for _ in range(3)]
R = [np.full((1002, 5), True) for _ in range(2)]

env = StandardBreakout()
env.reset()

feature_matrix = FeatureMatrix(env, attrs_num=4)

model = SchemaNetwork(W, R)
model.set_proxy_env(feature_matrix)

actions = model.plan_actions()

feature_matrix.planned_action = actions[0]
print(feature_matrix.planned_action)
