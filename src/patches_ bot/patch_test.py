from __future__ import print_function
from random import choice
from vizdoom import *
from time import sleep
from patch_bot import PatchBot
import cv2
import copy


game = DoomGame()

game.load_config("health_gathering_supreme.cfg")
# game.load_config("./health_gathering_supreme.cfg")

game.add_game_args("+name AI +colorset 0")

game.set_screen_format(ScreenFormat.BGR24)

game.set_mode(Mode.PLAYER)

game.set_window_visible(False)

game.init()

# Play with this many bots
bots = 5

# Run this many episodes
episodes = 10

last_health = 100
last_frags = 0
last_level_frags = 0

my_bot = PatchBot()
bot_frags = 0

for i in range(episodes):

    # Add specific number of bots
    # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
    # edit this file to adjust bots).
    game.send_game_command("removebots")
    for _ in range(bots):
        game.send_game_command("addbot")

    # Play until the game (episode) is over.
    while not game.is_episode_finished():

        state = game.get_state()
        screen = state.screen_buffer
        action = my_bot.update(screen, True, True)

        # Make action and check health
        game.make_action(action)
        # sleep(0.01)

        health = game.get_game_variable(GameVariable.HEALTH)
        if health != last_health:
            last_health = health
            print("Player health: " + str(health))

        frags = game.get_game_variable(GameVariable.FRAGCOUNT)
        if frags != last_frags:
            last_frags = frags
            print("Player has " + str(frags - last_level_frags) + " frags.")

        # Check if player is dead
        if game.is_player_dead():
            print("Player died.")
            # Use this to respawn immediately after death, new state will be available.
            my_bot.renew()
            game.respawn_player()

    print(my_bot.counter)

    print("Episode finished.")
    print("************************")

    print("Results:")
    server_state = game.get_server_state()
    for i in range(len(server_state.players_in_game)):
        if server_state.players_in_game[i]:
            if server_state.players_names[i] != "AI":
                bot_frags += server_state.players_frags[i]
            print(server_state.players_names[i] + ": " + str(server_state.players_frags[i]))
    print("************************")
    # break

    last_level_frags = last_frags

    # Starts a new episode. All players have to call new_episode() in multiplayer mode.
    game.new_episode()

print(frags)
print(bot_frags)
game.close()
