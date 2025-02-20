from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# measure="Energy"
measure="Wasserstein"

# nn_size="small"
nn_size="big"


Simulations=30
setting_list=["1", "2", "3"]
samplesize_list=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]




data=list()

for setting_index in setting_list:

    ebrm_inaccuracy_dict=defaultdict(list)
    qrdqn_inaccuracy_dict=defaultdict(list)
    mmdqn_inaccuracy_dict=defaultdict(list)

    for sample_size in samplesize_list:

        directory_name= "Results/" + nn_size + str(5) + "_sample" + str(sample_size)

        ebrm_inaccuracy_list_pergame=[]
        qrdqn_inaccuracy_list_pergame=[]
        mmdqn_inaccuracy_list_pergame=[]


        true_distribution=np.load("materials/setting"+setting_index+ '_N3000.npy')

        for sim_ind in range(1,Simulations+1):

            if measure=="Energy":
                ebrm_result = np.load(directory_name + "/ebrm/setting" + setting_index +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
                ebrm_inaccuracy=ebrm_result[1]
                ebrm_inaccuracy_list_pergame.append(ebrm_inaccuracy)

                qrdqn_result = np.load(directory_name + "/qrdqn/setting" + setting_index +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
                qrdqn_inaccuracy=qrdqn_result[1]
                qrdqn_inaccuracy_list_pergame.append(qrdqn_inaccuracy)

                mmdqn_result = np.load(directory_name + "/mmdqn/setting" + setting_index +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
                mmdqn_inaccuracy=mmdqn_result[1]
                mmdqn_inaccuracy_list_pergame.append(mmdqn_inaccuracy)

            elif measure=="Wasserstein":
                ebrm_result = np.load(directory_name + "/ebrm/setting" + setting_index +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
                qrdqn_result = np.load(directory_name + "/qrdqn/setting" + setting_index +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)
                mmdqn_result = np.load(directory_name + "/mmdqn/setting" + setting_index +"_seed"+str(sim_ind)+'.npy', allow_pickle=True)

                ebrm_particles=ebrm_result[0]
                qrdqn_particles=qrdqn_result[0]
                mmdqn_particles=mmdqn_result[0]

                ebrm_inaccuracy=wasserstein_distance(true_distribution, ebrm_particles)
                qrdqn_inaccuracy=wasserstein_distance(true_distribution, qrdqn_particles)
                mmdqn_inaccuracy=wasserstein_distance(true_distribution, mmdqn_particles)

                ebrm_inaccuracy_list_pergame.append(ebrm_inaccuracy)
                qrdqn_inaccuracy_list_pergame.append(qrdqn_inaccuracy)
                mmdqn_inaccuracy_list_pergame.append(mmdqn_inaccuracy)



        ebrm_inaccuracy_dict[sample_size]+=ebrm_inaccuracy_list_pergame
        qrdqn_inaccuracy_dict[sample_size]+=qrdqn_inaccuracy_list_pergame
        mmdqn_inaccuracy_dict[sample_size]+=mmdqn_inaccuracy_list_pergame


    methods = {
        "EBRM": ebrm_inaccuracy_dict,
        "QRDQN": qrdqn_inaccuracy_dict,
        "MMDQN": mmdqn_inaccuracy_dict
    }

    data_mean = {}
    for method, data_dict in methods.items():
        data_mean[method] = [np.mean(data_dict[size]) for size in samplesize_list]
    df_mean = pd.DataFrame(data_mean, index=samplesize_list)
    df_mean.index.name = "Sample Size"

    data_std = {}
    for method, data_dict in methods.items():
        data_std[method] = [np.std(data_dict[size]) for size in samplesize_list]
    df_std = pd.DataFrame(data_std, index=samplesize_list)
    df_std.index.name = "Sample Size"

    print("Setting "+setting_index)
    print("\n")
    print(np.round(df_mean.T,3))
    print(np.round(df_std.T,3))
    print("\n\n")

    data.append( (df_mean, df_std) )




### Plot

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(setting_list), figsize=(18, 6))

if len(setting_list) == 1:
    axes = [axes]

for idx, (df_mean, df_std) in enumerate(data):

    ax = axes[idx]

    for method in df_mean.columns:

        # Plot the mean line
        ax.plot(df_mean.index, df_mean[method], marker='o', label=method)

        # Calculate bounds for shaded areas
        mean = df_mean[method]
        std = df_std[method]
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std

        # Add shaded areas
        ax.fill_between(df_mean.index, lower_bound, upper_bound, alpha=0.2, linestyle='--', label='_nolegend_')

    ax.set_title(f"Setting {setting_list[idx]}", fontsize=14)
    ax.set_xlabel("Sample Size", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel("Inaccuracy", fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(setting_list), fontsize=10, title="Method")
fig.suptitle(measure + "-Inaccuracies in " + nn_size + "-NN model" , fontsize=16, y=0.95)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave space for the global legend
plt.savefig("Results/"+ measure + "_NN" + nn_size + ".png")

