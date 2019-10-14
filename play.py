from model.schemanet import SchemaNet
from model.player import Player


if __name__ == '__main__':
    model = SchemaNet()
    reward_model = SchemaNet()
    standard_player = Player(model, reward_model)
    standard_player.play(step_num=2, log=True)
