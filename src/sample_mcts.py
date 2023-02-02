import math
import random
from typing import Dict, Tuple, List, Optional

import torch
from torch import nn
from tqdm import tqdm

from game_mechanics import (
    transition_function,
    is_terminal,
    choose_move_randomly,
    play_go,
    reward_function, save_pkl,
)
from .node import Node, NodeID
from .utils import BLACK, BOARD_SIZE
from state import State

gpu = torch.device("cuda")
cpu = torch.device("cpu:0")

TOTAL_NUM_MOVES = BOARD_SIZE**2 + 1

my_max = max


class AGZMCTS:
    def __init__(
        self,
        initial_state: State,
        c_puct: float,
        network,
        eta_noise_root_node: float = 0.25,
        verbose: int = 0,
    ):
        self.root_node = Node(initial_state)

        # Maps node IDs to nodes
        self.tree: Dict[str:Node] = {self.root_node.key: self.root_node}
        self.total_return: Dict[str:float] = {self.root_node.key: 0.0}
        self.N: Dict[str:int] = {self.root_node.key: 0}

        # Get the neural network policy at the root node for use in PUCT
        v, pol = network(self.root_node)
        # TODO: You'll want to add to this, but it depends on your design choice
        ...

        self.c_puct = c_puct
        self.network = network

        self.verbose = verbose

    def do_rollout(self) -> None:
        if self.verbose:
            print("\nNew rollout started from", self.root_node.key)
        path_taken = self._select()
        chosen_node = self._expand(path_taken[-1])
        value = self._evaluate(chosen_node)
        self._backup(path_taken, chosen_node, value)

    def _select(self) -> List[NodeID]:
        """Selects a node to simulate from, given the current state and tree.

        Returns a list of nodes of the path taken from the root
         to the selected node.
        """
        node = self.root_node
        if self.verbose:
            print("Selecting from node:", node.key)
        path_taken = [node.key]

        while not node.is_terminal and node.key in self.tree:
            node_id = self._alphago_select(node)
            path_taken.append(node_id)

            if node_id not in self.tree:
                if self.verbose:
                    print("Node not in tree, expanding:", node_id)
                break
            # Update `node` object for next iteration
            node = self.tree[node_id]

        return path_taken

    def _expand(self, node_id: NodeID) -> Node:
        """Unless the selected node is a terminal state, expand the selected node by adding its
        children nodes to the tree.
        """
        if node_id in self.tree:
            return self.tree[node_id]

        if self.verbose:
            print("Expanding node:", node_id)

        parent_node = self.tree[node_id[:-1]]
        # Action taken to get to this node is the last element of the node ID
        new_state = transition_function(parent_node.state, node_id[-1].move)
        child_node = Node(new_state)
        self.tree[child_node.key] = child_node
        self.total_return[child_node.key] = 0
        self.N[child_node.key] = 0
        return child_node

    def _evaluate(self, node: Node) -> float:
        """
        Evaluates a node

        Want to use:
        ```
        value, policy = self.network(node)
        ```
        to get the value of `node`.

        Treat terminal states differently - use the reward function.

        Cache policy output so you can use it in future PUCT calculations.
        """
        ...
        raise NotImplementedError()

    def _backup(self, path_taken: List[NodeID], evaluated_node: Node, backup_value: float) -> None:
        """
        Update the action-value estimates of all parent nodes in the tree with the
        return from the simulated trajectory.

        Remember to consider whether values are relative to the current player or not.
         This is a design choice you make - what are you training your neural network
         to predict?
        """
        for node_id in path_taken:
            # Since values are in range [0, 1], inverse is 1 - value
            node = self.tree[node_id]
            node_backup_value = backup_value if node.state.to_play == evaluated_node.state.to_play else 1 - backup_value
            self.total_return[node_id] += node_backup_value
            self.N[node_id] += 1

            if self.verbose >= 2:
                print(
                    "Backing up node:",
                    node_id,
                    self.N[node_id],
                    self.total_return[node_id],
                )

    def choose_action(self, temperature: float) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Once we've simulated all the trajectories, we want to
         select the action at the current timestep which
         maximises the action-value estimate.
        """
        if self.verbose:
            print("N:", {a: self.N.get(state, 0) for a, state in self.root_node.child_node_ids.items()})

        if temperature == 0:
            # Use most-visited - do this when evaluating the agent
            return my_max(
                self.root_node.child_node_ids.keys(),
                key=lambda a: self.N[self.root_node.child_node_ids[a]],
            ), None

        # Fill in a lot here to calculate the action probabilities and sample from it
        output_probs = ...
        chosen_action = ...

        if self.verbose:
            print("Action probabilities:", output_probs)
            print("Chosen action:", chosen_action)
        return chosen_action, output_probs

    def Q(self, node_id: Tuple) -> float:
        return self.total_return[node_id] / (self.N[node_id] + 1e-15)

    def _alphago_select(self, parent: Node) -> NodeID:
        """Implement PUCT algorithm from AlphaGo Zero paper."""
        raise NotImplementedError()

    def prune_tree(self, new_root_state: State) -> None:
        """This is optional, but will lead to an improvement in performance.

        You can prune the tree to only include nodes which are relevant to the new state.
         I.e. only include all ancestors of the new root node.
        """
        raise NotImplementedError()