@echo off
@REM Asumption: We have run 1.py and 2_1.py.

SET PARTICLES=5
SET SIM=30
@REM SET SIM=3

SET NN_SIZE=small
@REM SET NN_SIZE=big

SET LR_TD=0.005
SET EBRM_ITERATIONS=3000
SET LR_EBRM=0.005

setlocal enabledelayedexpansion

for %%S in (100 200 300 400 500 600 700 800 900 1000) do (
    SET SAMPLE_SIZE=%%S
    
    if exist "Results\%NN_SIZE%%PARTICLES%_sample!SAMPLE_SIZE!" (
        rmdir /S /Q "Results\%NN_SIZE%%PARTICLES%_sample!SAMPLE_SIZE!"
        echo Directory removed: Results\%NN_SIZE%%PARTICLES%_sample!SAMPLE_SIZE!
    ) else (
        echo Directory does not exist: Results\%NN_SIZE%%PARTICLES%_sample!SAMPLE_SIZE!
    )

    echo.
    echo Playing setting1 with sample size !SAMPLE_SIZE!.
    echo.

    python 3_1_QRDQN_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting1 --sample_size !SAMPLE_SIZE! --n_quantile %PARTICLES% --lr %LR_TD% --target_update_freq 5 
    python 3_2_MMDQN_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting1 --sample_size !SAMPLE_SIZE! --n_particle %PARTICLES% --lr %LR_TD% --target_update_freq 5 
    python 3_3_EBRM_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting1 --sample_size !SAMPLE_SIZE! --particle_N %PARTICLES% --lr %LR_EBRM% --iterations %EBRM_ITERATIONS% 

    echo.
    echo Playing setting2 with sample size !SAMPLE_SIZE!.
    echo.

    python 3_1_QRDQN_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting2 --sample_size !SAMPLE_SIZE! --n_quantile %PARTICLES% --lr %LR_TD% --target_update_freq 5 
    python 3_2_MMDQN_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting2 --sample_size !SAMPLE_SIZE! --n_particle %PARTICLES% --lr %LR_TD% --target_update_freq 5 
    python 3_3_EBRM_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting2 --sample_size !SAMPLE_SIZE! --particle_N %PARTICLES% --lr %LR_EBRM% --iterations %EBRM_ITERATIONS% 

    echo.
    echo Playing setting3 with sample size !SAMPLE_SIZE!.
    echo.

    python 3_1_QRDQN_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting3 --sample_size !SAMPLE_SIZE! --n_quantile %PARTICLES% --lr %LR_TD% --target_update_freq 5 
    python 3_2_MMDQN_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting3 --sample_size !SAMPLE_SIZE! --n_particle %PARTICLES% --lr %LR_TD% --target_update_freq 5 
    python 3_3_EBRM_simulation.py --simulations %SIM% --nn_size %NN_SIZE% --setting setting3 --sample_size !SAMPLE_SIZE! --particle_N %PARTICLES% --lr %LR_EBRM% --iterations %EBRM_ITERATIONS% 

    echo.
    echo Running comparison for sample size !SAMPLE_SIZE!.
    echo.

    nohup python 4_1_simulation_comparison.py --simulations %SIM% --settings setting1 setting2 setting3 --nn_size %NN_SIZE% --particles %PARTICLES% --sample_size !SAMPLE_SIZE! >> Results/%NN_SIZE%%PARTICLES%_sample!SAMPLE_SIZE!/SIM_%NN_SIZE%%PARTICLES%_sample!SAMPLE_SIZE!.txt
)

endlocal