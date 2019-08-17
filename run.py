import numpy as np
from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
from model.inference import SchemaNetwork


W = [np.zeros((5, 5)) for _ in range(3)]
R = [np.zeros((5, 5)) for _ in range(2)]

env = StandardBreakout()
#env.reset()

feature_matrix = FeatureMatrix(env, attrs_num=4)

model = SchemaNetwork(W, R)
model.set_proxy_env(env)

actions = model.plan_actions()

feature_matrix.planned_action = actions[0]
