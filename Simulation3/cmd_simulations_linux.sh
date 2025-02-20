#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=small_samplesetA       #Set the job name to "JobExample4"
#SBATCH --time=1-00:00:00              #day-hr-min-sec
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=2G                  #Request 2GB per node
#SBATCH --output=job1.%j      #Send stdout/err to "Example4Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=hse1223@tamu.edu    #Send all emails to email_address 

#First Executable Line
module load GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.12.1-CUDA-11.7.0
cd $SCRATCH
source env1/bin/activate 
cd AISTATS_accepted

PARTICLES=5
SIM=30

NN_SIZE="small"
LR_TD=0.005
EBRM_ITERATIONS=3000
LR_EBRM=0.005

for SAMPLE_SIZE in 100 200 300 400
do
    if [ -d "Results/${NN_SIZE}${PARTICLES}_sample${SAMPLE_SIZE}" ]; then
        rm -rf "Results/${NN_SIZE}${PARTICLES}_sample${SAMPLE_SIZE}"
        echo "Directory removed: Results/${NN_SIZE}${PARTICLES}_sample${SAMPLE_SIZE}"
    else
        echo "Directory does not exist: Results/${NN_SIZE}${PARTICLES}_sample${SAMPLE_SIZE}"
    fi

    echo
    echo "Playing setting1 with sample size $SAMPLE_SIZE."
    echo

    python3 3_1_QRDQN_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting1 --sample_size $SAMPLE_SIZE --n_quantile $PARTICLES --lr $LR_TD --target_update_freq 5 
    python3 3_2_MMDQN_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting1 --sample_size $SAMPLE_SIZE --n_particle $PARTICLES --lr $LR_TD --target_update_freq 5 
    python3 3_3_EBRM_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting1 --sample_size $SAMPLE_SIZE --particle_N $PARTICLES --lr $LR_EBRM --iterations $EBRM_ITERATIONS 

    echo
    echo "Playing setting2 with sample size $SAMPLE_SIZE."
    echo

    python3 3_1_QRDQN_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting2 --sample_size $SAMPLE_SIZE --n_quantile $PARTICLES --lr $LR_TD --target_update_freq 5 
    python3 3_2_MMDQN_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting2 --sample_size $SAMPLE_SIZE --n_particle $PARTICLES --lr $LR_TD --target_update_freq 5 
    python3 3_3_EBRM_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting2 --sample_size $SAMPLE_SIZE --particle_N $PARTICLES --lr $LR_EBRM --iterations $EBRM_ITERATIONS 

    echo
    echo "Playing setting3 with sample size $SAMPLE_SIZE."
    echo

    python3 3_1_QRDQN_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting3 --sample_size $SAMPLE_SIZE --n_quantile $PARTICLES --lr $LR_TD --target_update_freq 5 
    python3 3_2_MMDQN_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting3 --sample_size $SAMPLE_SIZE --n_particle $PARTICLES --lr $LR_TD --target_update_freq 5 
    python3 3_3_EBRM_simulation.py --simulations $SIM --nn_size $NN_SIZE --setting setting3 --sample_size $SAMPLE_SIZE --particle_N $PARTICLES --lr $LR_EBRM --iterations $EBRM_ITERATIONS 

    echo
    echo "Running comparison for sample size $SAMPLE_SIZE."
    echo

    nohup python3 4_1_simulation_comparison.py --simulations $SIM --settings setting1 setting2 setting3 --nn_size $NN_SIZE --particles $PARTICLES --sample_size $SAMPLE_SIZE >> "Results/${NN_SIZE}${PARTICLES}_sample${SAMPLE_SIZE}/SIM_${NN_SIZE}${PARTICLES}_sample${SAMPLE_SIZE}.txt" 
done


