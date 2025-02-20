### .npy files

# Each .npy file in ebrm_results and qrdqn_results contains a tuple with 3 elements.
# 0-th element: estimated distribution with 3000 particles.
# 1-th element: Energy-inaccuracy
# 2-th element: computation time => But ebrm (ran 50,000 iterations) and qrdqn (ran 500,000 iterations) were implemented in different machines. So unfair comparison. Just care about the number of iterations.


##### PART1 : Energy-inaccuracy (pooled)

from collections import defaultdict
import numpy as np

SIM=100

games = ["acrobot", "cartpole", "mountaincar"]

ebrm_inaccuracy_list=defaultdict(list)
qrdqn_inaccuracy_list=defaultdict(list)
mmdqn_inaccuracy_list=defaultdict(list)


for game in games:

    ebrm_inaccuracy_list_pergame=[]
    qrdqn_inaccuracy_list_pergame=[]
    mmdqn_inaccuracy_list_pergame=[]

    for sim_ind in range(1,SIM+1):

        ebrm_result = np.load('ebrm_results/' + game +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        ebrm_inaccuracy=ebrm_result[1]
        ebrm_inaccuracy_list_pergame.append(ebrm_inaccuracy)

        qrdqn_result = np.load('qrdqn_results/' + game +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        qrdqn_inaccuracy=qrdqn_result[1]
        qrdqn_inaccuracy_list_pergame.append(qrdqn_inaccuracy)

        mmdqn_result = np.load('mmdqn_results/' + game +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        mmdqn_inaccuracy=mmdqn_result[1]
        mmdqn_inaccuracy_list_pergame.append(mmdqn_inaccuracy)


    ebrm_inaccuracy_list[game]+=ebrm_inaccuracy_list_pergame
    qrdqn_inaccuracy_list[game]+=qrdqn_inaccuracy_list_pergame
    mmdqn_inaccuracy_list[game]+=mmdqn_inaccuracy_list_pergame


print('\n\nEnergy-inaccuracy (marginalized)\n')
for game in games:

    print(game + ":"+'\n')
    print('EBRM: Mean={:.4f}, STD={:.4f}'.format(np.mean(ebrm_inaccuracy_list[game]), np.std(ebrm_inaccuracy_list[game])))
    print('QRDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(qrdqn_inaccuracy_list[game]), np.std(qrdqn_inaccuracy_list[game])))
    print('MMDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(mmdqn_inaccuracy_list[game]), np.std(mmdqn_inaccuracy_list[game])))
    print("\n")



##### PART2 : Wasserq-inaccuracy (pooled)

import numpy as np
from scipy.stats import wasserstein_distance
from collections import defaultdict

SIM=50

games = ["acrobot", "cartpole", "mountaincar"]

ebrm_inaccuracy_list=defaultdict(list)
qrdqn_inaccuracy_list=defaultdict(list)
mmdqn_inaccuracy_list=defaultdict(list)


for game in games:

    ebrm_inaccuracy_list_pergame=[]
    qrdqn_inaccuracy_list_pergame=[]
    mmdqn_inaccuracy_list_pergame=[]


    if game == "cartpole":
        gamma=0.90; epsilon=0.90; skip=1
    elif game == "mountaincar":
        gamma=0.90; epsilon=0.30; skip=10
    elif game == "acrobot":
        gamma=0.90; epsilon=0.10; skip=10
    else:
        assert "game should be one of the three."

    true_distribution=np.load("materials/"+game+"_gamma"+str(int(gamma*100)) + "eps" + str(int(epsilon*100)) + "skip" + str(skip) + 'N3000.npy')

    for sim_ind in range(1,SIM+1):

        ebrm_result = np.load('ebrm_results/' + game +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        qrdqn_result = np.load('qrdqn_results/' + game +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
        mmdqn_result = np.load('mmdqn_results/' + game +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)

        ebrm_particles=ebrm_result[0]
        qrdqn_particles=qrdqn_result[0]
        mmdqn_particles=mmdqn_result[0]

        ebrm_inaccuracy=wasserstein_distance(true_distribution, ebrm_particles)
        qrdqn_inaccuracy=wasserstein_distance(true_distribution, qrdqn_particles)
        mmdqn_inaccuracy=wasserstein_distance(true_distribution, mmdqn_particles)

        ebrm_inaccuracy_list_pergame.append(ebrm_inaccuracy)
        qrdqn_inaccuracy_list_pergame.append(qrdqn_inaccuracy)
        mmdqn_inaccuracy_list_pergame.append(mmdqn_inaccuracy)

    ebrm_inaccuracy_list[game]+=ebrm_inaccuracy_list_pergame
    qrdqn_inaccuracy_list[game]+=qrdqn_inaccuracy_list_pergame
    mmdqn_inaccuracy_list[game]+=mmdqn_inaccuracy_list_pergame




print('\n\nWasserstein1-inaccuracy (marginalized)\n')
for game in games:
    
    print(game + ":"+'\n')
    print('EBRM: Mean={:.4f}, STD={:.4f}'.format(np.mean(ebrm_inaccuracy_list[game]), np.std(ebrm_inaccuracy_list[game])))
    print('QRDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(qrdqn_inaccuracy_list[game]), np.std(qrdqn_inaccuracy_list[game])))
    print('MMDQN: Mean={:.4f}, STD={:.4f}'.format(np.mean(mmdqn_inaccuracy_list[game]), np.std(mmdqn_inaccuracy_list[game])))
    print("\n")


