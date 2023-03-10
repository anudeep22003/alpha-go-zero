import pickle
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pygame
import torch
from gym.spaces import Box, Discrete
from pygame import Surface
from torch import nn

from go_base import (
    BLACK,
    BOARD_SIZE,
    PASS_MOVE,
    WHITE,
    all_legal_moves,
    game_over,
    int_to_coord,
    is_move_legal,
    play_move,
    result,
    score,
)
from render import render_game
from state import State

HERE = Path(__file__).parent.resolve()

# Hack as this won't pip install on replit
sys.path.append(str(HERE / "PettingZoo"))


ALL_POSSIBLE_MOVES = np.arange(BOARD_SIZE**2 + 1)

# Visuals
SCREEN_SIZE = (500, 500)

# The komi to use is much debated. 7.5 seems to
# generalise well for different board sizes
# lifein19x19.com/viewtopic.php?f=15&t=17750
# 7.5 is also the komi used in alpha-go vs Lee Sedol
# (non-integer means there are no draws)

KOMI = 7.5


def transition_function(state: State, action: int) -> State:
    return play_move(state, action, state.to_play)


def reward_function(state: State) -> int:
    if not game_over(state.recent_moves):
        return 0
    result_ = result(state.board, KOMI)
    return result_ if state.player_color == BLACK else result_ * -1


def is_terminal(state: State) -> bool:
    return game_over(state.recent_moves)


def play_go(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier: float = 1.0,
    render: bool = True,
    verbose: bool = False,
) -> float:

    env = GoEnv(
        opponent_choose_move,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    state, reward, done, info = env.reset()
    done = False
    while not done:
        action = your_choose_move(state=state)
        state, reward, done, info = env.step(action)
    return reward


# TODO: Currently state.board is just relative to the colors (BLACK = 1, WHITE = -1)
# Need to think about whether to change before giving to choose_moves


class GoEnv:
    def __init__(
        self,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
    ):

        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        # Which color do we play as

        self.state = State()

        if render:
            self.init_visuals()

    def init_visuals(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Go")
        self._render_game()

    def _render_game(
        self,
    ) -> None:
        render_game(self.state.board, screen=self.screen)

    @property
    def reward(self) -> int:
        return reward_function(self.state)

    @property
    def done(self) -> bool:
        return is_terminal(self.state)

    def reset(self) -> Tuple[State, float, bool, Dict]:

        # 1 is black and goes first, white is -1 and goes second
        self.player_color = random.choice([BLACK, WHITE])
        self.color_str = "Black" if self.player_color == BLACK else "White"

        self.state = State(player_color=self.player_color)

        if self.verbose:
            print(
                f"Resetting Game.\nYou are playing with the {self.color_str} tiles.\nBlack plays first\n\n"
            )
            print(self.state)

        if self.state.to_play != self.player_color:
            self._step(
                self.opponent_choose_move(state=self.state),
            )

        return self.state, self.reward, self.done, {}

    def move_to_string(self, move: int) -> str:
        N = self.state.board.shape[0]
        if move == N**2:
            return "passes"
        return f"places counter at coordinate: {(move//N, move%N)}"

    def __str__(self) -> str:
        return str(self.state.board) + "\n"

    def _step(self, move: int) -> None:

        if self.verbose:
            name = "player" if self.state.to_play == self.player_color else "opponent"
            print(f"{name} {self.move_to_string(move)}")
            print('--- state before stepping ----')
            print(self.state)

        assert not self.done, "Game is done! Please reset() the env before calling step() again"
        assert is_move_legal(
            int_to_coord(move), self.state.board, self.state.ko
        ), f"{move} is an illegal move"

        self.state = transition_function(self.state, move)

        if self.verbose:
            print('--- state after stepping ----')
            print(self.state)
            
        if self.render:
            self._render_game()

    def step(self, move: int) -> Tuple[State, int, bool, Dict]:

        assert self.state.to_play == self.player_color
        self._step(move)

        if not self.done:
            self._step(self.opponent_choose_move(state=self.state))

        if self.verbose:
            print('--------- full step -----------')
            print(self.state)

        if self.verbose and self.done:
            self.nice_prints()  # Probably not needed

        return self.state, self.reward, self.done, {}

    def nice_prints(self):
        print(
            f"\nGame over. Reward = {self.reward}.\n"
            f"Player was playing as {self.color_str}\n"
            f"Black has {np.sum(self.state.board==1)} counters.\n"
            f"White has {np.sum(self.state.board==-1)} counters.\n"
            f"Your score is {score(self.state.board, KOMI)}.\n"
        )


def choose_move_randomly(state: State) -> int:
    return random.choice(all_legal_moves(state.board, state.ko))


def choose_move_pass(state: State) -> int:
    """Always pass."""
    return PASS_MOVE


def load_pkl(team_name: str, network_folder: Path = HERE) -> nn.Module:
    net_path = network_folder / f"{team_name}_file.pkl"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    with open(net_path, "rb") as handle:
        file = pickle.load(handle)
    return file


def save_pkl(file: Any, team_name: str) -> None:
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_file.pkl"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            with open(net_path, "wb") as handle:
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            load_pkl(team_name)
            return
        except Exception:
            if attempt == n_retries - 1:
                raise


# Need to know the default screen size from petting zoo to get which square is clicked
# Will not work with a screen override
SQUARE_SIZE = SCREEN_SIZE[0] // BOARD_SIZE
LEFT = 1


def pos_to_coord(pos: Tuple[int, int]) -> Tuple[int, int]:  # Assume square board
    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE
    return row, col


def coord_to_int(coord: Tuple[int, int]) -> int:
    return coord[0] * BOARD_SIZE + coord[1]


def human_player(state) -> int:

    print("Your move, click to place a tile!")
    legal_moves = all_legal_moves(state.board, state.ko)
    if len(legal_moves) == 1:
        print("You have no legal moves, so you pass")
        return legal_moves[0]

    while True:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONUP and event.button == LEFT:
                coord = pos_to_coord(pygame.mouse.get_pos())
                action = coord_to_int(coord)
                if action in legal_moves:
                    return action
