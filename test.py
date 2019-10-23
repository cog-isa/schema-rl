from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
from model.inference import SchemaNetwork
from testing.testing import W, R

env = StandardBreakout()
env.reset()

model = SchemaNetwork(W, R, [None, None])
model.set_curr_iter(0)

actions = model.plan_actions()

print('Number of actions planned: {}'.format(actions.size))
print(actions)
