import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network model with specified hyperparameters.")
    parser.add_argument("--simulations", type=int, help="Number of simulations to run.")
    parser.add_argument("--nn_size", type=str, choices=["big", "small"], help="Only two choices are available: big or small.")
    parser.add_argument("--setting", type=str, choices=["setting1", "setting2", "setting3", "setting4"], help="Only two choices are available: big or small.")
    parser.add_argument("--sample_size", type=int, help="Number of samples.")
    parser.add_argument("--iterations", type=int, help="Number of iterations.")    
    parser.add_argument("--particle_N", type=int, help="Number of particles.")
    parser.add_argument("--lr", type=float, help="learning rate.")
    return parser.parse_args()


##### STEP 1: IMPORT LIBRARIES

import torch
from torch import nn
import numpy as np
import random
import gym
from collections import deque
import itertools
import time
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 


##### STEP 2: LOAD THE OPTIMAL POLICY NETWORK

REPORTING_FREQ = 500; display_size=300;  
args = parse_args()
Simulations=args.simulations
NN_size=args.nn_size
BUFFER_SIZE=args.sample_size
particle_N=args.particle_N
LR=args.lr
setting=args.setting
Iterations=args.iterations
# BATCH_SIZE=BUFFER_SIZE
BATCH_SIZE=min(BUFFER_SIZE, 100)

directory_name = "Results/" + NN_size+str(particle_N)+"_sample"+str(BUFFER_SIZE) + "/ebrm"
os.makedirs(directory_name, exist_ok=True)


env=gym.make('CartPole-v1')
if setting=="setting1":
    gamma=0.90; epsilon=0.90; skip=1; done_rew = -5. # setting-1: left-skewed
elif setting=="setting2":
    gamma=0.90; epsilon=0.90; skip=5; done_rew = -5. # setting-2: bell-shape
elif setting=="setting3":
    gamma=0.90; epsilon=0.70; skip=1; done_rew = -5. # setting-3: concentrated
else:
    assert "setting not appropriate."

action_num=env.action_space.n
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

optimal_net = Network(env)
optimal_net.to(device)
optimal_net.load_state_dict(torch.load("materials/cartpole_optimal_model.pt"))


##### STEP 3: Neural Network

class EBRMNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__() 
        self.device = device

        if NN_size=="small":
            self.network = nn.Sequential(
                nn.Linear(input_dim, 10),
                nn.ReLU(),
                nn.Linear(10, action_num*particle_N) 
            )
        elif NN_size=="big":
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_num*particle_N) 
            )     

    def forward(self, x):
        return self.network(x)
     
    def compute_loss(self, transitions):

        states = [t[0] for t in transitions]
        actions = [t[1] for t in transitions]
        rews = [t[2] for t in transitions]
        state_post = [t[3] for t in transitions]

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device) 
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device)     
        state_post_t = torch.as_tensor(state_post, dtype=torch.float32, device=self.device)

        ## LHS distribution

        eval_q = self(states_t)
        eval_q = eval_q.reshape(BATCH_SIZE, action_num, particle_N)  # batch * action * quantiles
        eval_q = torch.stack([eval_q[i][actions_t[i]] for i in range(BATCH_SIZE)]) # batch * quantiles

        ## RHS distribution

        target_actions=[]
        for i in range(BATCH_SIZE):
            rnd_sample = random.random() # uniform(0,1)
            if rnd_sample <= epsilon:
                action = env.action_space.sample()
            else:
                action= optimal_net.act(state_post_t[i])
            target_actions.append(action)


        next_q=self(state_post_t)            # IMPORTANT! NO DETACH

        next_q=next_q.reshape(BATCH_SIZE, action_num, particle_N)
        next_q=torch.stack([next_q[i][target_actions[i]] for i in range(BATCH_SIZE)]) # take the target action    
        target_q=rews_t.unsqueeze(-1) + gamma * next_q

        LHS = eval_q.unsqueeze(2) # mb_size x n_particle x 1 
        RHS = target_q.unsqueeze(2) # mb_size x n_particle x 1 

        LHS_subtr_LHS_abs = torch.abs(LHS - LHS.transpose(1,2))
        LHS_subtr_RHS_abs = torch.abs(LHS - RHS.transpose(1,2))
        RHS_subtr_RHS_abs = torch.abs(RHS - RHS.transpose(1,2))

        ## Compute loss - assuming N=M, but efficient.
        k_LL = -LHS_subtr_LHS_abs.unsqueeze(3) 
        k_LR = -LHS_subtr_RHS_abs.unsqueeze(3) 
        k_RR = -RHS_subtr_RHS_abs.unsqueeze(3) 
        k_sums = k_LL + k_RR - 2 * k_LR
        MMD_loss_bybatch = k_sums.sum(dim=3).mean(dim=(1,2)) 
        MMD_loss_bybatch[MMD_loss_bybatch < 0] = 0. 
        MMD_loss = MMD_loss_bybatch.mean()

        return MMD_loss



##### SIMULATION 

seedvec=[i for i in range(1,Simulations+1)] 
a=1221423; b=1242351

for seedno in seedvec:

    seed=seedno*a+b

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)    # seed before intializing
    env.action_space.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    online_net=EBRMNN(input_dim=env.observation_space.shape[0])
    online_net.to(device)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)


    ### STAGE 1: Collect Data

    online_net.train()
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    obs=env.reset()
    for Time in range(BUFFER_SIZE):

        action = env.action_space.sample()
        for _ in range(skip):
            new_obs, rew, done, _ = env.step(action)   # repeat the same action for given time frames.
            if done:
                rew=done_rew
                env.seed(1)           # seed after reaching terminal state
                new_obs=env.reset()    
                break

        transition=(obs, action, rew, new_obs)
        replay_buffer.append(transition)
        obs=new_obs


    ### STAGE 2: Train OPE model.

    MC_returns_original = np.load("materials/"+setting + "_N3000.npy")
    X_subtr_X = np.expand_dims(MC_returns_original, axis=1) - np.expand_dims(MC_returns_original, axis=0)  
    X_subtr_X = np.abs(X_subtr_X)

    start_time=time.time()

    for Time in range(Iterations):

        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time=time.time()


    ### STAGE 3: FINAL INACCURACY

    online_net.eval()

    DISPLAY_SIZE = 3000

    ## Estimated distribution

    random.seed(seed)
    env.seed(seed)   # seed before initializing
    env.action_space.seed(seed) 

    obses_t=[]
    actions=[]
    for i in range(DISPLAY_SIZE):
        obs=env.reset()
        obs_t=torch.tensor(obs, dtype=torch.float32, device=device)
        obses_t.append(obs_t)
        rnd_sample = random.random() # uniform(0,1)
        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action= optimal_net.act(obs)
        actions.append(action)

    obses_t=torch.stack(obses_t)

    histogram=[]
    target_q_temp = online_net(obses_t).reshape(DISPLAY_SIZE, action_num, particle_N).detach()

    index_list = [random.randint(0, particle_N-1) for _ in range(DISPLAY_SIZE)]
    for i in range(DISPLAY_SIZE):
        candidates=target_q_temp[i][actions[i]]
        value=candidates[index_list[i]].item()
        histogram.append(value)


    ## Energy Inaccuracy

    Estimate_returns_original=np.array(histogram)
    Y_subtr_Y = np.expand_dims(Estimate_returns_original, axis=1) - np.expand_dims(Estimate_returns_original, axis=0)
    Y_subtr_Y = np.abs(Y_subtr_Y)
    X_subtr_Y = np.expand_dims(MC_returns_original, axis=1) - np.expand_dims(Estimate_returns_original, axis=0)
    X_subtr_Y = np.abs(X_subtr_Y)
    energy_inaccuracy = 2*np.mean(X_subtr_Y) - np.mean(X_subtr_X) - np.mean(Y_subtr_Y)  

    cumulative_seconds = end_time - start_time
    np.save(directory_name + "/" + setting + '_seed'+str(seedno), (Estimate_returns_original, energy_inaccuracy, cumulative_seconds))

    print('-----------------')
    print('seed={:d}: Final Energy-Inaccuracy={:4f}'.format(seedno, energy_inaccuracy))
    print('Time: {:4f} seconds spent.'.format(cumulative_seconds))
    print('-----------------')
    print('\n')


