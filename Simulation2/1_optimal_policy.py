### https://www.youtube.com/watch?v=NP8pXZdU-5U&list=LL&index=1&t=1435s  => youtube lecture
### https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py  => description of the setting in CartPole (e.g.) state & action space
### https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py  => description of the setting in MountainCar (e.g.) state & action space
### recommended to install gym=0.17.3 (pip install gym=0.17.3). Otherwise, there becomes extra {} in obs when a new episode begins.


##### STEP 1: IMPORT LIBRARIES

import torch
from torch import nn
import numpy as np
import random
import gym
from collections import deque
import itertools
import time

##### STEP 2: TUNING PARAMETERS

# gamma=0.99
gamma=0.999

batch_size=32
min_replay_size=1000 # when |D| gets bigger than this, start training.
buffer_size=50000 # maximum level of |D|

epsilon_start=1.0
epsilon_end=0.02
epsilon_decay=10000

target_update_freq=1000
max_step = 300000

game="cartpole"
game="mountaincar"
game="acrobot"
# game="pendulum"



if game == "cartpole":
    env=gym.make('CartPole-v1')
elif game == "mountaincar":
    env=gym.make('MountainCar-v0')
elif game == "acrobot":
    env=gym.make('Acrobot-v1')
else:
    assert "game should be one of the three."


##### STEP 3: CREATE NETWORK

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

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

replay_buffer=deque(maxlen=buffer_size) 
return_buffer=deque([0.0], maxlen=100)   
episode_return=0.0

online_net = Network(env)
online_net.to(device)
target_net = Network(env)
target_net.to(device)
target_net.load_state_dict(online_net.state_dict()) 

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4) 


##### STEP 4: INITIALIZE REPLAY BUFFER

obs=env.reset()

for _ in range(min_replay_size): 
    
    action = env.action_space.sample() 
    new_obs, rew, done, _ = env.step(action) 
    transition = (obs, action, rew, done, new_obs) 
    replay_buffer.append(transition) 
    obs=new_obs

    if done:
        obs=env.reset() 


##### STEP 5: TRAINING LOOP

obs=env.reset()
step=0

iter=0 # If we want to repeat this traning loop, start from here.
for step in itertools.count(start=step): 

    start_time=time.time()

    ## (Part1) One step of action by epsilon-greedy policy

    epsilon = np.interp(step, [0, epsilon_decay], [epsilon_start, epsilon_end]) 
    # From a line that connects (0, epsilon_start) and (epsilon_decay, epsilon_end), we interpolate. 
    # Gives us the bound values when x-value is out of the bound (outside [0, epsilon_decay]).
    rnd_sample = random.random() # can be understood as uniform(0,1)

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action= online_net.act(obs)

    new_obs, rew, done, _ = env.step(action)
    # a=env.render() # may comment out if we want to speed up the training by not showing the animation.
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition) # plug new tuple of (s, a, reward, s_prime) into our replay_buffer D.

    obs=new_obs
    episode_return += rew  
    # Of course this is not the actual return that our learning (gamma=0.99) is targetting, since this is assuming gamma=1.0 in this case.
    # However, if we learn with gamma=1, it will be more prone to diverge, so we should use gamma=0.99 when learning.
    # Nevertheless, what we are really interested in is how long the agent persists, so we evaluate it with gamma=1.0.
    
    if done:
        obs=env.reset()
        return_buffer.append(episode_return) # Every time we are done with an episode, we obtain one return. (but with limit of return_buffer size)
        episode_return=0


    ## (Part2) Sample Batch

    transitions = random.sample(replay_buffer, batch_size)

    obses = np.asarray([t[0] for t in transitions]) 
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions]) 
    
    obses_t = torch.as_tensor(obses, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32, device=device).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=device)

    ## (Part3) Compute Targets, Loss, and Gradient Descent

    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] 
    targets = rews_t + gamma * (1-dones_t) * max_target_q_values
    
    q_values = online_net(obses_t) 
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t) 
    loss=nn.functional.smooth_l1_loss(action_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    ## (Part4) Update Target Network

    if step % target_update_freq==0:
        target_net.load_state_dict(online_net.state_dict()) 

        end_time=time.time()

        print('Step', step)
        print('Avg Return', np.mean(return_buffer))
        print('Time: {:4f} seconds spent.'.format(end_time - start_time))

    iter += 1
    if iter-1==max_step:
        break



##### STEP 6: SAVE
    
online_net.to(torch.device('cpu'))
torch.save(online_net.state_dict(), "materials/"+ game + '_optimal_model.pt')


##### STEP 7: DISPLAY THE TRAINED MODEL

obs=env.reset()
while True: # Stop with Ctrl+C. When doing it, we need to click the terminal, and no part should be dragged (highlighted). Then Ctrl+C works.
    action=online_net.act(obs)
    obs, _, done, _ = env.step(action)
    a=env.render()
    if done:
        obs=env.reset()



