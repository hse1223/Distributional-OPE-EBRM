### .npy files

# Each .npy file in ebrm_results and qrdqn_results contains a tuple with 3 elements.
# 0-th element: estimated distribution with 3000 particles.
# 1-th element: Energy-inaccuracy
# 2-th element: computation time => But ebrm (ran 50,000 iterations) and qrdqn (ran 500,000 iterations) were implemented in different machines. So unfair comparison. Just care about the number of iterations.


##### PART1 : Energy-inaccuracy (marginal)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network model with specified hyperparameters.")
    parser.add_argument("--simulations", type=int, help="Number of simulations to run.")
    parser.add_argument("--settings", type=str, nargs="+", help="settings that are tested.")
    parser.add_argument("--nn_size", type=str, choices=["big", "small"], help="Only two choices are available: big or small.")
    parser.add_argument("--particles", type=int, help="Number of particles.")
    parser.add_argument("--sample_size", type=int, help="Number of samples.")
    return parser.parse_args()

args = parse_args()
Simulations=args.simulations
settings=args.settings
directory_name= "Results/" + args.nn_size + str(args.particles) + "_sample" + str(args.sample_size)

from collections import defaultdict
import numpy as np

ebrm_inaccuracy_list=defaultdict(list)
qrdqn_inaccuracy_list=defaultdict(list)
mmdqn_inaccuracy_list=defaultdict(list)


for setting in settings:

    ebrm_inaccuracy_list_pergame=[]
    qrdqn_inaccuracy_list_pergame=[]
    mmdqn_inaccuracy_list_pergame=[]

    for sim_ind in range(1,Simulations+1):

        ebrm_result = np.load(directory_name + "/ebrm/" + setting +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        ebrm_inaccuracy=ebrm_result[1]
        ebrm_inaccuracy_list_pergame.append(ebrm_inaccuracy)

        qrdqn_result = np.load(directory_name + "/qrdqn/" + setting +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        qrdqn_inaccuracy=qrdqn_result[1]
        qrdqn_inaccuracy_list_pergame.append(qrdqn_inaccuracy)

        mmdqn_result = np.load(directory_name + "/mmdqn/" + setting +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        mmdqn_inaccuracy=mmdqn_result[1]
        mmdqn_inaccuracy_list_pergame.append(mmdqn_inaccuracy)


    ebrm_inaccuracy_list[setting]+=ebrm_inaccuracy_list_pergame
    qrdqn_inaccuracy_list[setting]+=qrdqn_inaccuracy_list_pergame
    mmdqn_inaccuracy_list[setting]+=mmdqn_inaccuracy_list_pergame


print('\n\nEnergy-inaccuracy (marginalized)\n')
for setting in settings:

    print(setting + ":"+'\n')
    print('EBRM: Mean={:.4f}, STD={:.4f}'.format(np.mean(ebrm_inaccuracy_list[setting]), np.std(ebrm_inaccuracy_list[setting])))
    print('QRDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(qrdqn_inaccuracy_list[setting]), np.std(qrdqn_inaccuracy_list[setting])))
    print('MMDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(mmdqn_inaccuracy_list[setting]), np.std(mmdqn_inaccuracy_list[setting])))
    print("\n")



##### PART2 : Wasserq-inaccuracy (marginal)

import numpy as np
from scipy.stats import wasserstein_distance
from collections import defaultdict

ebrm_inaccuracy_list=defaultdict(list)
qrdqn_inaccuracy_list=defaultdict(list)
mmdqn_inaccuracy_list=defaultdict(list)


for setting in settings:

    ebrm_inaccuracy_list_pergame=[]
    qrdqn_inaccuracy_list_pergame=[]
    mmdqn_inaccuracy_list_pergame=[]

    if setting=="setting1":
        gamma=0.90; epsilon=0.90; skip=1; done_rew = -5. # setting-1: left-skewed
    elif setting=="setting2":
        gamma=0.90; epsilon=0.90; skip=5; done_rew = -5. # setting-2: bell-shape
    elif setting=="setting3":
        gamma=0.90; epsilon=0.70; skip=1; done_rew = -5. # setting-3: concentrated
    else:
        assert "setting not appropriate."

    true_distribution=np.load("materials/"+setting+'_N3000.npy')

    for sim_ind in range(1,Simulations+1):

        ebrm_result = np.load(directory_name + "/ebrm/" + setting +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        qrdqn_result = np.load(directory_name + "/qrdqn/" + setting +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        mmdqn_result = np.load(directory_name + "/mmdqn/" + setting +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)

        ebrm_particles=ebrm_result[0]
        qrdqn_particles=qrdqn_result[0]
        mmdqn_particles=mmdqn_result[0]

        ebrm_inaccuracy=wasserstein_distance(true_distribution, ebrm_particles)
        qrdqn_inaccuracy=wasserstein_distance(true_distribution, qrdqn_particles)
        mmdqn_inaccuracy=wasserstein_distance(true_distribution, mmdqn_particles)

        ebrm_inaccuracy_list_pergame.append(ebrm_inaccuracy)
        qrdqn_inaccuracy_list_pergame.append(qrdqn_inaccuracy)
        mmdqn_inaccuracy_list_pergame.append(mmdqn_inaccuracy)

    ebrm_inaccuracy_list[setting]+=ebrm_inaccuracy_list_pergame
    qrdqn_inaccuracy_list[setting]+=qrdqn_inaccuracy_list_pergame
    mmdqn_inaccuracy_list[setting]+=mmdqn_inaccuracy_list_pergame




print('\n\nWasserstein1-inaccuracy (marginalized)\n')
for setting in settings:
    
    print(setting + ":"+'\n')
    print('EBRM: Mean={:.4f}, STD={:.4f}'.format(np.mean(ebrm_inaccuracy_list[setting]), np.std(ebrm_inaccuracy_list[setting])))
    print('QRDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(qrdqn_inaccuracy_list[setting]), np.std(qrdqn_inaccuracy_list[setting])))
    print('MMDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(mmdqn_inaccuracy_list[setting]), np.std(mmdqn_inaccuracy_list[setting])))
    print("\n")


