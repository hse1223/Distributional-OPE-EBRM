seedvec=[i for i in range(1,101)] 
a=1221423; b=1242351

game="acrobot"
# game="cartpole"
# game="mountaincar"

BUFFER_SIZE=100; n_particle=5; TARGET_UPDATE_FREQ = 5; LR=1e-2; BATCH_SIZE=1; display_size=300; 

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
os.environ['KMP_DUPLICATE_LIB_OK']='True' # If I use 


##### STEP 2: LOAD THE OPTIMAL POLICY NETWORK

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
            nn.Linear(64, action_num) 
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


##### STEP 4: Neural Network

class MMDQN(nn.Module):
    def __init__(self, input_dim):
        super().__init__() 

        self.device = device

        self.network = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, action_num*n_particle) 
        )


    def forward(self, x):
        return self.network(x)
     
    def compute_loss(self, transitions, target_net):

        ## Batch

        states = [t[0] for t in transitions]
        actions = [t[1] for t in transitions]
        rews = [t[2] for t in transitions]
        state_post = [t[3] for t in transitions]

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device) # needs to be a column vector
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device)     # needs to be a column vector
        state_post_t = torch.as_tensor(state_post, dtype=torch.float32, device=self.device)


        ## LHS distribution

        eval_q = self(states_t)
        eval_q = eval_q.reshape(BATCH_SIZE, action_num, n_particle)  # batch * action * quantiles
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


        next_q=target_net(state_post_t).detach()
        next_q=next_q.reshape(BATCH_SIZE, action_num, n_particle)
        next_q=torch.stack([next_q[i][target_actions[i]] for i in range(BATCH_SIZE)]) # take the target action
    
        target_q=rews_t.unsqueeze(-1) + gamma * next_q


        LHS = eval_q.unsqueeze(2) # mb_size x n_particle x 1 
        RHS = target_q.unsqueeze(2) # mb_size x n_particle x 1 

        LHS_subtr_LHS_abs = torch.abs(LHS - LHS.transpose(1,2))
        LHS_subtr_RHS_abs = torch.abs(LHS - RHS.transpose(1,2))
        RHS_subtr_RHS_abs = torch.abs(RHS - RHS.transpose(1,2))

        ## Compute loss - assuming N=M, but efficient.
        k_LL = -LHS_subtr_LHS_abs
        k_LR = -LHS_subtr_RHS_abs 
        k_RR = -RHS_subtr_RHS_abs 
        k_sums = k_LL + k_RR - 2 * k_LR
        MMD_loss_bybatch = k_sums.mean(dim=(1,2))
        MMD_loss_bybatch[MMD_loss_bybatch < 0] = 0.
        MMD_loss = MMD_loss_bybatch.mean()

        return MMD_loss


##### SIMULATION 

for seedno in seedvec:

    seed=seedno*a+b

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)      # seed before initializing
    env.action_space.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    online_net=MMDQN(input_dim=env.observation_space.shape[0])
    online_net.to(device)
    target_net=MMDQN(input_dim=env.observation_space.shape[0])
    target_net.to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)


    ### STAGE 1: Collect Data

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    obs=env.reset()
    for Time in range(BUFFER_SIZE):
        action = env.action_space.sample()
        for _ in range(skip):
            new_obs, rew, done, _ = env.step(action)   # repeat the same action for given time frames. (altered the game)
            if done:
                rew=done_rew
                env.seed(1)           
                new_obs=env.reset()   # seed after reaching terminal state
                break 
        transition=(obs, action, rew, new_obs)
        replay_buffer.append(transition)
        obs=new_obs


    ### STAGE 2: Train OPE model.
    MC_returns_original = np.load("materials/"+game+"_gamma"+str(int(gamma*100))+"eps"+str(int(epsilon*100)) + "skip" + str(skip) + "N3000.npy")
    X_subtr_X = np.expand_dims(MC_returns_original, axis=1) - np.expand_dims(MC_returns_original, axis=0) 
    X_subtr_X = np.abs(X_subtr_X)

    Time_total=0
    random.shuffle(replay_buffer)
    start_time=time.time()

    for Time in range(BUFFER_SIZE):

        Time_total+=1

        transition=replay_buffer.pop()
        loss = online_net.compute_loss([transition], target_net) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (Time+1) % TARGET_UPDATE_FREQ == 0:

            target_net.load_state_dict(online_net.state_dict())
            target_net.eval()


    end_time=time.time()


    ### STAGE 3: FINAL INACCURACY

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
    target_q_temp = target_net(obses_t).reshape(DISPLAY_SIZE, action_num, n_particle)      
    index_list = [random.randint(0, n_particle-1) for _ in range(DISPLAY_SIZE)]
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
    np.save('mmdqn_results/' + game + '_seed'+str(seedno), (Estimate_returns_original, energy_inaccuracy, cumulative_seconds))

    print('-----------------')
    print('seed={:d}: Final Energy-Inaccuracy={:4f}'.format(seedno, energy_inaccuracy))
    print('Time: {:4f} seconds spent.'.format(cumulative_seconds))
    print('-----------------')
    print('\n')


