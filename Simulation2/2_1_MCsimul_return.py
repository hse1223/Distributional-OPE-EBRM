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

setting="setting1"
# setting="setting2"
# setting="setting3"

# Simulation=3000; display=False # used in measuring simulation inaccuracy
Simulation=100000; display=False # used in displaying for 2_2.py

env=gym.make('CartPole-v1')
if setting=="setting1":
    gamma=0.90; epsilon=0.90; skip=1; done_rew = -5. # setting-1: left-skewed
elif setting=="setting2":
    gamma=0.90; epsilon=0.90; skip=5; done_rew = -5. # setting-2: bell-shape
elif setting=="setting3":
    gamma=0.90; epsilon=0.70; skip=1; done_rew = -5. # setting-3: concentrated
else:
    assert "setting not appropriate."

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network(nn.Module):
    def __init__(self, env): # => optimal_net = Network(env)
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
        q_values = self(obs_t) # for each state, there can be two actions.
        max_q_index = torch.argmax(q_values) # select the maximizing action. 
        action = max_q_index.item()
        return action


optimal_net = Network(env)
optimal_net.to(device)
optimal_net.load_state_dict(torch.load("materials/cartpole_optimal_model.pt"))


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

        rnd_sample = random.random() # epsilon-greedy using uniform(0,1)
        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action= optimal_net.act(obs)
        for _ in range(skip):        # repeat the same action for given time frames. (altered the game)
            new_obs, rew, done, _ = env.step(action)  

            if done:
                rew=done_rew
                env.seed(1)          # seed after reaching terminal state
                new_obs=env.reset()   
                break 

        reward_vec[iter] = rew
        obs=new_obs


    gammas = np.array([gamma**i for i in range(max_iter)])
    y_return = sum(reward_vec * gammas)
    Y_vector[episode]=y_return

    print('Episode = {:d}/{:d} Y={:4f}'.format(episode+1, Simulation, y_return))


import os
os.makedirs('materials', exist_ok=True)
if not display:
    np.save("materials/"+setting + "_N" + str(Simulation), Y_vector)



##### STEP 5: PLOT
    
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

plt.hist(Y_vector, bins=50, density=True, color='blue', alpha=0.7)
plt.xlabel('Return')
plt.ylabel('probability')
plt.title('Return (marginal) in ' + setting)
plt.grid(True)

if display:
    plt.show()
else:
    plt.savefig("materials/"+setting + "_N" + str(Simulation) + ".jpg", format="jpg") 
    plt.close()



