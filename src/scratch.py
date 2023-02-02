import random
import numpy as np
from go_base import all_legal_moves
from game_mechanics import play_go, choose_move_randomly, human_player, transition_function
from state import State
from mcts import Node, MCTS

# play_go(
#     your_choose_move=human_player,
#     opponent_choose_move=choose_move_randomly,
#     game_speed_multiplier=1,
#     render=True,
#     verbose=True,
# )

s = State()
n = Node(s)

print(n.state)

ax = [0,1,2,3,4]
for a in ax: 
    print(f"step {a}")
    s = transition_function(n.state,a)
    n = Node(s)
    print(n.state)
