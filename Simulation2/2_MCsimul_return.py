##### STEP 1: IMPORT LIBRARIES

import torch
from torch import nn
import numpy as np
import random
import gym
from collections import deque
import itertools
import time


##### STEP 2: LOAD NETWORK

# game="acrobot"
# game="cartpole"
game="mountaincar"

Simulation=3000; display=False
# Simulation=500; display=True

if game == "cartpole":
    env=gym.make('CartPole-v1'); done_rew = -5.
    gamma=0.90; epsilon=0.90; skip=1
elif game == "mountaincar":
    env=gym.make('MountainCar-v0'); done_rew = 2.
    gamma=0.90; epsilon=0.30; skip=10
elif game == "acrobot":
    env=gym.make('Acrobot-v1'); done_rew = 5.
    gamma=0.90; epsilon=0.10; skip=10
else:
    assert "game should be one of the three."

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### 3.1: nn.Module

class Network(nn.Module):
    def __init__(self, env): 
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape)) 
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), 
            nn.Tanh(),
            nn.Linear(64, env.action_space.n) 
        )
        self.device=device

    def forward(self, x): 
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_values = self(obs_t) 
        max_q_index = torch.argmax(q_values) 
        action = max_q_index.item()
        return action


optimal_net = Network(env)
optimal_net.to(device)
optimal_net.load_state_dict(torch.load("materials/"+game+'_optimal_model.pt'))


##### STEP 3: DISPLAY THE TRAINED MODEL

# obs=env.reset()
# while True: # Stop with Ctrl+C. When doing it, we need to click the terminal, and no part should be dragged (highlighted). Then Ctrl+C works.

#     rnd_sample = random.random() # uniform(0,1)
#     if rnd_sample <= epsilon:
#         action = env.action_space.sample()
#     else:
#         action= optimal_net.act(obs)

#     # action=optimal_net.act(obs)
#     obs, _, done, _ = env.step(action)
#     a=env.render()
#     if done:
#         obs=env.reset()



##### STEP 4: MC SIMULATION

optimal_net.eval() 

max_iter = int(np.emath.logn(gamma, 1e-3))  # so that gamma^{max_iter} = 1e-3 (very small)

Y_vector = np.zeros(Simulation)
for episode in range(Simulation):

    env.seed(episode); random.seed(episode) # seed before initializing
    env.action_space.seed(episode)

    reward_vec = np.zeros(max_iter)
    obs=env.reset()

    for iter in range(max_iter):

        rnd_sample = random.random() # uniform(0,1)
        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action= optimal_net.act(obs)
        for _ in range(skip):
            new_obs, rew, done, _ = env.step(action)   # repeat the same action for given time frames. (altered the game)

            if done:
                rew=done_rew
                env.seed(1)          # seed after reaching terminal state
                new_obs=env.reset()   
                break 

        reward_vec[iter] = rew
        obs=new_obs
        # a=env.render() # To see how it performs with eps-greedy. 

    gammas = np.array([gamma**i for i in range(max_iter)])
    y_return = sum(reward_vec * gammas)
    Y_vector[episode]=y_return

    print('Episode = {:d}/{:d} Y={:4f}'.format(episode+1, Simulation, y_return))


### Save

np.save("materials/"+game+"_gamma"+str(int(gamma*100))+"eps"+str(int(epsilon*100)) + "skip" + str(skip) + "N" + str(Simulation), Y_vector)



##### STEP 5: PLOT
    
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

Y_vector=np.load("materials/"+game+"_gamma"+str(int(gamma*100))+"eps"+str(int(epsilon*100)) + "skip" + str(skip) + "N" + str(Simulation) + ".npy")
plt.hist(Y_vector, bins=50, density=True, color='blue', alpha=0.7)
plt.xlabel('Return')
plt.ylabel('probability')
plt.title('Histogram')
plt.grid(True)

if display:
    plt.show()
else:
    plt.savefig("materials/"+game+"_gamma"+str(int(gamma*100))+"eps"+str(int(epsilon*100)) + "skip" + str(skip) + "N" + str(Simulation) + ".jpg") 
    plt.close()



