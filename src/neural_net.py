import torch 
import torch.nn as nn
from collections import deque
import numpy as np

# create two headed neural net 

# hyperparams
IN_CHANNELS = 18    #! change this later to 16
INPUTS = 82
BOARD_SIZE = 9
OUTPUT_ACTIONS = BOARD_SIZE**2 + 1


class PVNetSimple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s = nn.Sequential(
            nn.Conv2d(in_channels=IN_CHANNELS, out_channels=48, kernel_size=1, stride=1),      # shared layers
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),
        )
        
        self.v = nn.Sequential(
            nn.Conv2d(48,20,2,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),
            nn.Flatten(),
            nn.Linear(80,1)
        )

        self.p = nn.Sequential(
            nn.Conv2d(48,20,2,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),
            nn.Flatten(),
            nn.Linear(80,BOARD_SIZE**2+1),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x_batch):
        x = self.s(x_batch)
        p = self.p(x)
        v = self.v(x)
        return p,v
    
class PVNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # conv layer 1
        self.s1 = nn.Conv2d(in_channels=IN_CHANNELS, out_channels=24, kernel_size=3, stride=3)      # shared layers
        self.b1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        
        # conv layer 2
        self.s2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=3)               # policy layer
        self.b2 = nn.BatchNorm2d(48)

        # conv layer 3
        self.s3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=3)               # value layer
        self.b3 = nn.BatchNorm2d(64)

        # conv layer 4
        self.s4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=3)               # value layer
        self.b4 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        # policy layer 
        self.p = nn.Sequential(
            nn.Linear(400,300),
            nn.ReLU(),
            nn.Linear(300,200),
            nn.ReLU(),
            nn.Linear(200, OUTPUT_ACTIONS),
            nn.Softmax()
        )

        # value layer 
        self.v = nn.Sequential(
            nn.Linear(400,200),
            nn.ReLU(),
            nn.Linear(500,00),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def shared_forward(self, x):
        c1 = self.s1(x)
        c1 = self.b1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool_1(c1)
#       c1 = self.maxpool_1(self.relu(self.b1(self.s1(x))))
        c2 = self.maxpool_1(self.relu(self.b2(self.s2(c1))))
        c3 = self.maxpool_1(self.relu(self.b3(self.s3(c2)) + c1))         # residual layer
        c4 = self.maxpool_1(self.relu(self.b4(self.s4(c3)) + c2))         # residual layer
        return self.flatten(c4)
        
    def policy_forward(self, shared_x):
        return self.p(shared_x)
    
    def value_forward(self, shared_x):
        return self.v(shared_x)
    
    def forward(self, x):
        shared_x = self.shared_forward(x)
        policy_probabilities = self.policy_forward(shared_x)
        value = self.value_forward(shared_x)
        return policy_probabilities, value


class ChannelConstructor:
    def __init__(self) -> None:
        self.self_memory = deque(maxlen=8)
        self.opponents_memory = deque(maxlen=8)
    
    def get_empty_board(self):
        return torch.zeros((BOARD_SIZE,BOARD_SIZE))

    def initialize_empty_boards(self):
        for _ in range(8):
            self.self_memory.append(self.get_empty_board())
    