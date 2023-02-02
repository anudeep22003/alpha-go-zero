# Alpha Go Zero Implementation 

### Intent
To learn how the reinforcement learning algorithm works

### Details
- Implemented a lot of this in Colab to take advantage of the GPU compute. Pulled out the pkl file which is a checkpoint that I then used for running the final code in pygame.
- The final version is a complete implementation, but due to lack of Millions in compute, the behaviour is only slightly better than random 
- However, this is a complete implementation of the algorithm taking advantage of many concepts at the edge of RL. The concepts implemented are listed in the next section.

### Concepts implemented
(While all these are not directly implemented in Alpha Go Zero, the final concepts built on all the earlier ones. Hence listing here for completeness.)
- Markov Decision 
- Generalized policy iteration 
- Q Learning
- Policy Gradient in continuous action space
- TD Lambda
- lambda-return 
- Actor Critic Methods
- Generalized Advantage Estimation
- Proximal Policy Optimization 
	- Clipped PPO
- Model based RL and Planning
- Monte Carlo Tree Search 
- APV-MCTS (Asynchronous Policy Value and Monte Carlo Tree Search)

### Time taken
- This was implemented over a 2 month period. 
- I participated in a series of weekly games where the rl-bots I made would compete with others'

### Credits 
- Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction
- Delta Academy; https://joindeltaacademy.com/


