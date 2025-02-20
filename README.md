# Distributional-OPE-EBRM
These codes are for the paper &lt;Distributional Off-policy Evaluation with Bellman Residual Minimization> that is accepted by AISTATS 2025.
In total, this contains three separate simulations.

## Simulation-1: Non-completeness (Section 5.2)
- We have designed a tabular setting that does not satisfy completeness assumption.
- We compared this with FLE (and QRTD).
- We used R.
- We included all the simulation results in the directory. You can delete those if you want to start by yourself.

## Simulation-2: OpenAI-gym (Section 5.1)
- We have tried three different games (acrobot, cartpole, mountaincar), all of which use 100 samples.
- We compared EBRM with two different deep neural network based methods, QRDQN and MMDQN.
- We used Python.
- We included all the simulation results in the directory. You can delete those if you want to start by yourself.


## Simulation-3: Cartpole (Appendix D.2.2)
- Among the OpenAI-gym games, we did more intense simulations on cartpole, which seems to best satisfy realizability assumption.
- We have compared with QRDQN and MMDQN, based on more various sample sizes and also trying a larger neural network.
- We used Python.
- We did not include the simulation results this time, so that the users could try for themselves.


