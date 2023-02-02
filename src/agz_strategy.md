### Approach:
- [ ] create two output headed network (value function estimate v_θ and policy output p_θ) 
  - [ ] initial layers will be sequence of CNN layers 
  - [ ] subsequent layers will be independent hidden layers 
- [ ] Craft the input array to have 17 channels 
  - [ ] 8 past player boards
  - [ ] 8 past opponent boards
  - [ ] 1s or 0s depending on black or white to play next 

### MCTS Algorithm 
- [ ] select
  - [ ] u term which is combination of value estimate and puct value
    - [ ] cache the p-value from prev runs of the multi-head output for speed up
- [ ] expand and evaluate 
  - [ ] no need to run rollout policy, estimate value using v(θ) ) 
    - [ ] At this point, the forward pass through the neural net will generate both v(θ) and p(θ). We only use the v(θ). Cache the p(θ) value for later use.
- [ ] backup
  - [ ] straighforward use of value estimate form prior step to back up all ancestor nodes
  - [ ] action selection 
    - [ ] will incorporate the temperature parameter to do a stochastic sampling 
    - [ ] ### -----> ask Henry how to do this <-------

#### Node Class
- [ ] the MCTS algorithm needs a node class
- [ ] use recent_moves along with the board as the key 

### Loss function 
- [ ] v(θ) loss
  - [ ] MSE loss between value estimate and Monte Carlo return 
    - [ ] loss = (z_t - v_theta(state))^2 
- [ ] p(θ) loss 
  - [ ] cross entropy loss between network policy and MCTS policy 
- [ ] add l2 regularization 

### Look into 

- [ ] legal action masking 
- [ ] you have to store the number of visits to every node in a lookup table in MCTS
- [ ] maintain a deque of 400 and when th ewin rate of the last 400 games is higher than `55%` then save the network state at that point as a checkpoint and play against it 
- [ ] add batch normalization after every layer 
- [ ] add residual block (past value is propogated to future layer) 

### Questions
- [ ] what are channels in the input, are they additional computation on the board i.e a reformulation of the board? 
- [ ] do we update the 2-headed network after every step or after every episode (both p and v networks). Am i right in understanding that you update p at every step, update v at end of episode?

### HyperParams
- [ ] temperature
  - [ ] for first 30 moves temperature = 1, 
  - [ ] add dirchlet noise to the root node of tree to encourage exploration
- [ ] momentum parameter to control the adam optimizer 
- [ ] regularization parameter

