from collections import namedtuple
import math
import time
from random import random
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from neural_net import BOARD_SIZE
from state import State
from game_mechanics import is_terminal, transition_function, reward_function
from go_base import all_legal_moves
from utils import PlayerMove, BLACK, WHITE

NodeID = Tuple[PlayerMove, ...]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu:0")

"""
########### notes ########### 
value network 
- loss function = mean squared error (mcts value - value network value)

policy network
- loss fn = cross entropy (policy_probs - mcts_probs)
############################# 
"""


class Node:
    #! add channels into this 
    def __init__(self, state: State): 
        self.state = state
        self.is_terminal = is_terminal(state)

        # legal moves mask
        # returns an array of (board_size)**2+1 with 0 for illegal moves, and 1 for legal moves
        self.legal_moves_mask = torch.zeros((1, BOARD_SIZE ** 2 + 1,), device=device)
        self.legal_moves = torch.from_numpy(all_legal_moves(self.state.board, self.state.ko))
        self.legal_moves_mask[0, self.legal_moves] = 1

        self.children_ids: Dict[int, NodeID] = self._get_possible_children()   # legal children from this state 

    @property
    def key(self):
        """
        generates a key by taking advantage of the recent_moves function.
        - recent_moves: List[PlayerMoves,..]
        - PlayerMove: Tuple[int, int]
        since tuple is immutable, can be uses as a hash for the node_id

        """
        return self.state.recent_moves
    
    def _get_possible_children(self):
        """
        each legal action can create a possible child 
        the children are denoted by their node_ids
        so instead of 
        """
        if self.is_terminal:
            return {}
        else:
            # possible children ids is the current nodes key concatenated with a possible action
            #### debugging
            # print(' ------- getting possible children ---------')
            # print("node state", self.state, "\nlegal_moves", self.legal_moves)
            # print(' --- printing possible actions one by one -------')
            # for action in self.legal_moves:
            #     print("A --> ", action)
            print(" ---------------- new node initialized ---------------------")
            print(self.state)

            return {
                int(action): self.key + (PlayerMove(self.state.to_play, int(action)),) 
                for action in self.legal_moves
                }

def node_cnn_loader(node: Node):
    # see play_move definition to construct 16 channels
    cur_board = node.state.board
    player_past_8_boards = []
    opponent_past_8_boards = []
    player = node.state.player_color
    # print("all deltas ", type(node.state.board_deltas), node.state.board_deltas)
    for delta in node.state.board_deltas:
        # print("cur_board", cur_board.shape, "delta", delta.shape)
        player_board = np.where(cur_board==player,1,0)
        opp_board = np.where(cur_board==-player,1,0)
        player_past_8_boards.append(player_board)
        opponent_past_8_boards.append(opp_board)
        cur_board-=np.squeeze(delta, axis=0)
        
    # print("player_past_8_boards", player_past_8_boards)
    l = len(player_past_8_boards)
    if l<8:
        for _ in range(8-l):
            player_past_8_boards.append(np.zeros((BOARD_SIZE,BOARD_SIZE)))
            opponent_past_8_boards.append(np.zeros((BOARD_SIZE,BOARD_SIZE)))
    indication_channel = np.ones((BOARD_SIZE,BOARD_SIZE)) if node.state.to_play == BLACK else np.zeros((BOARD_SIZE,BOARD_SIZE))
    overall_channel = [node.state.board] + player_past_8_boards+opponent_past_8_boards+[indication_channel]
    channels = torch.stack([torch.tensor(c) for c in overall_channel])
    # print("channels_shape", channels.shape)
    
    return torch.unsqueeze(channels.float(), dim=0)

class MCTS:
    """
    step by step:
    1. start from root node and select a following node. 
        This step should return a path from the root node to the last node that is not in tree
        - use the value of the node, 
    """
    def __init__(self, 
#               starting_state: State,
                network: nn.Module,
                temperature: int,
                c_puct: float=5,
                verbose: bool = False,
                debug: bool = False,
                ):
        """You can use this as an mcts class that persists across choose_move calls."""
#       self.root_node = Node(starting_state)
        self.net = network

        ############# Dicts ############# 
        # contains a mapping of nodeid to node for all nodes ever visit 
        self.tree = {} 

        # mapping of nodeid to number of times node was visited
        # used to calculate probabilities and action selection 
        self.N: Dict[NodeID, int] = {}

        # mapping fron nodeid to the accumulated returns 
        # used to train the net and in action selection
        self.V: Dict[NodeID, float] = {}

        # mapping from nodeid to estimated_value, estimated_probabilites
        # used to prevent a second forward pass for subsequent use of policy probabilities
        self.cache: Dict[NodeID, torch.Tensor] = {}
        
        # training memory
        # this is where you will sample from to update the network 
        # NodeID: value, mcts_probabilities
        self.policy_memory: Dict[NodeID, torch.Tensor] = {}

        ################################### 

        # path taken through the game, keeps track of the active tree
        self.game_path_tree: list[NodeID] = []


        # relevant parameters to run the mcts algo
        self.c_puct = c_puct 
        self.temperature = temperature
        self.num_of_iterations = 200
        
        # operational flags
        self.verbose = verbose 
        self.debug = debug 


    def initialize_dicts(self, new_root_state: Node):
        if self.debug:
            print("-"*20)
            print(new_root_state)
            print("-"*20)
        # this is where to perform the mcts rollouts from
        self.root_node = Node(new_root_state)
        with torch.no_grad():
            pol_probs, val = self.net(node_cnn_loader(self.root_node))

        self.tree[self.root_node.key] = self.root_node
        self.V[self.root_node.key] = val
        self.N[self.root_node.key] = 0
        self.cache[self.root_node.key] = pol_probs
        self.game_path_tree.append(self.root_node.key)


    def do_mcts_rollout_to_choose_action(self, new_root_state: State, num_of_rollouts: int):
        self.initialize_dicts(new_root_state)
        # perform mcts computation to identify the action to take 
        # from the root node (which changes after every time step)
        if self.debug:
            print("-"*15, " Master Rollouts ", "-"*15, '\n\n')
            print("root node" , self.root_node.key)
            print("root node state" , self.root_node.state)
        self.do_rollouts(num_of_rollouts)
        # at this point the policy for the node that the mcts is being carried out is ready
        # use the root nodes children, look up their N values and choose depending on temperature
        chosen_action, policy_probs = self.choose_action()
        self.policy_memory[self.root_node.key] = policy_probs
        return chosen_action 

        
        
        
    def do_rollouts(self, num_roullouts: int):
        for rnum in range(num_roullouts):
            if self.debug:
                print("-"*8, " Sub-Rollouts ", "-"*8)
                print("#"*30, f"Rollout {rnum}", "#"*30)
            path_taken = self._select()
            expanded_node = self._expand(path_taken)
            value_to_backup = self._evaluate(expanded_node)
            self._backup(path_taken, expanded_node, value_to_backup)
        print("-"*10, " Rollouts completed ", "-"*10)
        
        
    def _select(self):
        # game tree's most recent value is the current root node to run rollout from
        node, node_id = self.root_node, self.root_node.key
        rollout_path_taken = [node_id]
        # choose an action if the state is not terminal and the node itself is in the tree
        # if not in the tree then this is a new node, add it to the tree and return the path
        # path will contain nodeids till the node not seen in the tree is encountered
        while not node.is_terminal and node.key in self.tree:
            action = self._agz_select_action(node)

            # use the node child_id mapping to get nodeid
            node_id = node.children_ids[int(action)]
            rollout_path_taken.append(node_id)
            if self.debug:
                print("agz selected action", action)
            if node_id not in self.tree:

                if self.verbose or self.debug:
                    print(f"encountered new node: {node_id}. \n Returning the path and expanding in next step.")
                break

            node = self.tree[node_id]

            if self.debug:
                if node.is_terminal:
                    print(f"encountered Terminal node. Returning path, and ending select.")
                else:
                    print(f"Already encountered: {node_id}, \n continuing...")
                time.sleep(0.5)

        return rollout_path_taken 

    def _agz_select_action(self, parent_node: Node) -> np.int32:
        #! modify to do uct selection
        if self.debug:
            print("parent_node_state",parent_node.state)
        all_action_values = torch.zeros(BOARD_SIZE**2+1)
        all_possible_actions = list(parent_node.children_ids.keys())
        # if self.debug:
        #     print("all possible actions", len(all_possible_actions), all_possible_actions)
        #     print("cached policy", self.cache[parent_node.key].shape, self.cache[parent_node.key])
        #     print("N --> \n", self.N)
        
        uct_val_for_possible_actions = torch.tensor([self.Q(node_id) + self.U(action, parent_node.key, node_id)
                                      for action, node_id
                                      in parent_node.children_ids.items()])
        all_action_values[all_possible_actions] = uct_val_for_possible_actions+1e-4

        if self.debug:
            print("all_possible_action", all_possible_actions)
            print("uct_val_for_possible_action", uct_val_for_possible_actions)
            print("all_action_values", all_action_values)
            print("cache policy probs", self.cache[parent_node.key])
        
        return torch.argmax(all_action_values).item()
    
    def Q(self, node_id: NodeID) -> float:
        return self.V.get(node_id,0) / (self.N.get(node_id,0) + 1e-15)
    
    def U(self, action: int, parent_node_id: NodeID, child_node_id: NodeID) -> float:
        u_val = (self.c_puct * self.cache[parent_node_id][0,action]) * math.sqrt(self.N[parent_node_id])/(1+self.N.get(child_node_id, 0))
        if self.debug:
            print("U value", u_val, "action: ", action, "parent; ", parent_node_id, "child", child_node_id)
        return u_val
    
    def _expand(self, path_taken: list):
        # if the last nodeid is in the tree then it is a terminal state, else
        # the last nodeid appended to path is a node that does not exist in the tree
        # add it to the tree, by generating a new node
        expand_node_id = path_taken[-1]

        # terminal state case
        if expand_node_id in self.tree:
            return self.tree[expand_node_id]
        
        
        # parent_node is the second to last nodeid
        parent_node_id = path_taken[-2]
        parent_node = self.tree[parent_node_id]

        if self.debug:
            print(f"received path ==> {path_taken}")
            print(f"nodeid of parent: {parent_node_id}")
            print(f"nodeid of node to expand: {expand_node_id}")
            print(f"PlayerMove to transition to new state: {expand_node_id[-1]}")
            time.sleep(1)

        # the node_id of the node to be expanded contains the player move at the end
        expanded_state = transition_function(parent_node.state, action=expand_node_id[-1].move)
        expanded_node = Node(expanded_state)

        # register this node to the tree
        # initialize count and return to 0
        self.tree[expand_node_id] = expanded_node
        self.V[expand_node_id] = 0.0
        self.N[expand_node_id] = 0


        return expanded_node
    
    def _evaluate(self, expanded_node: Node):
        # if state is terminal, take the reward and backpropogate it
        if expanded_node.is_terminal:
            val = reward_function(expanded_node.state)
            """
            We are optimizing for the opponent's win probability, or value of opponents board
            
            Case:
            - if black is to play and the reward is +1 
                - then it is black's victory 
                - however, since we are optimizing for opponent's win, we flip the reward
                - hence value to backup is -1 
            - if white is to play and the reward is -1 
                - then also it is black's victory, going by what we are optimizing for 
                - val to backup = +1  
            """
            if val == 1:
                if expanded_node.state.player_color == BLACK:
                    reward = -1
                else:
                    reward = +1
            elif val == -1:
                if expanded_node.state.player_color == BLACK:
                    reward = +1
                else:
                    reward = -1

            return reward

        # we run the new node through the neural net, and get action_probabilities and value of state 
        # we backup the state in the next step, we cache the probability when action is to be chosen in the next iteration
        pol_probs, val = self.net(node_cnn_loader(expanded_node))
        self.cache[expanded_node.key] = pol_probs

        if self.debug:
            if expanded_node.is_terminal:
                print(f"node is terminal, player color: {expanded_node.state.player_color} and to play: {expanded_node.state.to_play}")
                print("Achieved reward", reward)
            else:
                print(f"policy_type: {type(pol_probs)} \nactual_probs: {pol_probs} shape: {pol_probs.shape}")
            time.sleep(1)
        
        return val.item()

    def _backup(self, path_taken: Tuple[NodeID], expanded_node: Node, backup_value: float):
        # update the visit counts and value 
        # note: the backup value depends on node.state.to_play == BLACK or WHITE
        # if the node to be updated's player_color == expanded_node's player color then update flipped value
        # if it is an opposite node, then update as is
        # we do this because we are optimizing for opponents win 
        if self.debug:
            print("backup path length", len(path_taken))
        for node_id in path_taken:
            node = self.tree[node_id]
            #! slightly confused here, should it not be -backup_value?
            #! hence maybe take tanh as activation function to keep values b/w -1,1
            node_backup_value = backup_value if node.state.to_play != expanded_node.state.player_color else -backup_value
            self.N[node_id]+=1
            self.V[node_id]+=node_backup_value
            
            if self.debug:
                print(f"Node updated: {node.key} \n value_backed up: {node_backup_value}")
                time.sleep(1)
    
    def choose_action(self):
        # use most visited state in competitive play
        temperature = self.temperature
        if temperature == 0:
            action = max(
                self.root_node.children_ids.keys(),
                key= lambda a: self.N[self.root_node.children_ids[a]]
            )
            return action, None
        all_possible_actions = list(self.root_node.children_ids.keys())
        temperature_adjusted_counts = torch.tensor([self.N.get(node_id, 0) for _, node_id in self.root_node.children_ids.items()])**(1/temperature)

        # probabilities over all possible actions (incl masked)
        policy_probs = torch.zeros(BOARD_SIZE**2+1) 
        policy_probs[all_possible_actions] = temperature_adjusted_counts/torch.sum(temperature_adjusted_counts)
        
        if self.debug:
            print("all_possible_actions", all_possible_actions)
            print("temperature_adjusted_counts", temperature_adjusted_counts)
            print("policy probs", policy_probs)
        policy_distribution = Categorical(policy_probs)
        action = policy_distribution.sample()
        
        if self.debug:
            print("Action Probabilites: ", policy_distribution)
            print("Action: ", action.item())
        
        return action, policy_probs

        
    def get_game_memory(self, black_outcome: bool): # -> list(Tuple[State, torch.tensor, int]):
        mem = []
        for node_id in self.game_path_tree:
            node = self.tree[node_id]
            z = black_outcome if node.state.to_play != node.state.player_color else -black_outcome
            #! make sure policy size is correct
            mem.append((node_cnn_loader(node), self.policy_memory[node.key], z))