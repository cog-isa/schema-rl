from model.schemanet import SchemaNet

from player import Player

if __name__ == '__main__':
    model = SchemaNet()
    debug = False
    reward_model = SchemaNet(is_for_reward=True)
    standard_player = Player(model, reward_model)
    if debug:
        standard_player.play(log=False, cheat=False)
    else:
        standard_player.play(log=False, cheat=False)
        #try:
        #    standard_player.play(log=True, cheat=False)
        #except Exception:
        #    standard_player.save()
