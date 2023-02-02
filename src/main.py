from typing import Any, Dict, Optional
import random

from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch
from torch.optim import Adam 


# from check_submission import check_submission
from game_mechanics import (
    choose_move_randomly, 
    load_pkl, 
    play_go, 
    save_pkl, 
    human_player, 
    GoEnv, 
    transition_function, 
    is_terminal,
    reward_function
)
from state import State
from go_base import all_legal_moves
from neural_net import PVNet, PVNetSimple
from mcts import MCTS, Node

import torch

TEAM_NAME = "Anudeep"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


# hparams
EPOCHS = 2
BATCH_SIZE = 256
C_VALUE = 1e-4
LR = 1e-5
NUM_OF_ROLLOUTS = 1
TEMPERATURE = 4


env = GoEnv(
    opponent_choose_move=choose_move_randomly,
    verbose=True,
    render=False,
    game_speed_multiplier=1
    )

network = PVNetSimple()
optimizer = torch.optim.Adam(network.parameters(), lr=LR, weight_decay=C_VALUE)

def train() -> Any:

    # game opeational stuff
    memory = []

    """
    TODO: Write this function to train your algorithm.

    Returns:
        A pickleable object to be made available to choose_move
    """
    for ep in tqdm(range(EPOCHS)):
        # start state
        cur_game_state = State()
        mcts = MCTS(network, TEMPERATURE)
        done = 0
        step = 0
        while not done:
            print("-"*50)
            action = mcts.do_mcts_rollout_to_choose_action(new_root_state=cur_game_state, num_of_rollouts=NUM_OF_ROLLOUTS) 
            s_state = transition_function(cur_game_state, int(action))
            done = is_terminal(s_state)
            print(f"step {step} completed")
            print("action --> ", action)
            step+=1
            cur_game_state = s_state
        black_outcome = reward_function(state)
        memory+=mcts.get_game_memory(black_outcome)
        print(f"game #{ep} over, memory_size: {len(memory)}")

        # update the network at the end of every episode
        if len(memory)>BATCH_SIZE:
            # update here
            update_network(network, memory)
        
    return network
        
def update_network(network: nn.Module, memory, optimizer = optimizer):
    # create a batch
    batch = random.sample(memory, BATCH_SIZE)
    x,y,z = zip(*batch)
    
    inputs = torch.stack(x)
    pol_targets = torch.stack(y)
    val_targets = torch.unsqueeze(torch.tensor(z), dim=-1)
    
    pol_predictions, val_predictions = network(inputs)
    
    val_loss = nn.MSELoss(val_targets,val_predictions)
    pol_loss = nn.CrossEntropyLoss(pol_targets, pol_predictions)
    loss = val_loss + pol_loss 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    

# def choose_move(
#     state: State,
#     pkl_file: Optional[Any] = None,
#     mcts: Optional[MCTS] = None,
# ) -> int:
#     """Called during competitive play.
#      It returns a single action to play.

#     Args:
#         state: The current state of the go board (see state.py)
#         pkl_file: The pickleable object you returned in train
#         env: The current environment

#     Returns:
#         The action to take
#     """
#     legal_moves = all_legal_moves(state.board, state.ko)
# #   raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    # Example workflow, feel free to edit this! ###
    file = train()
    save_pkl(file, TEAM_NAME)

    my_pkl_file = load_pkl(TEAM_NAME)
    # my_mcts = MCTS()

    # Choose move functions when called in the game_mechanics expect only a state
    # argument, here is an example of how you can pass a pkl file and an initialized
    # mcts tree
    def choose_move_no_network(state: State) -> int:
        """The arguments in play_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
#       return choose_move(state, my_pkl_file, mcts=my_mcts)
        return choose_move_randomly

#    check_submission(
#        TEAM_NAME, choose_move_no_network
#    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # Play a game against against your bot!
    play_go(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_randomly,
        game_speed_multiplier=1,
        render=True,
        verbose=True,
    )
