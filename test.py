from environment.schema_games.breakout.games import StandardBreakout
from model.inference import SchemaNetwork
from testing.testing import TestFSS2

env = StandardBreakout()
env.reset()

W, R = TestFSS2.W, TestFSS2.R
model = SchemaNetwork(W, R, [None, None])
model.set_curr_iter(0)

actions = model.plan_actions()

print('Number of actions planned: {}'.format(actions.size))
print(actions)
