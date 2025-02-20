rm(list=ls())
# dev.off()

##### Parameter Setting

a=54325347; b=4326463

### Environment
gamma = 0.99
mu_last = 0
gridno = 100


### Setting

S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; seed=1 # setting2 - original
C1=1 # Muti-step EBRM
mstep_vec = c(2:10)

N = 500; Trials = c(1:50)
# N = 1000; Trials = c(1:50)
# N = 2000; Trials = c(1:50)
# N = 5000; Trials = c(1:50)
# N = 10000; Trials = c(1:25)
# N = 10000; Trials = c(26:50)
# N = 20000; Trials = c(1:17)
# N = 20000; Trials = c(18:34)
# N = 20000; Trials = c(35:50)


# S_size=30; A_true = 100; sig2_true = 5000; p_true=0.9; seed=1 # setting2: reasonable setting
# C1=1 # Muti-step EBRM
# Trials = 50
# mstep_vec = c(2:10)
# 
# N = 2000; Trials = c(1:50)
# N = 5000; Trials = c(1:50)
# N = 10000; Trials = c(1:50)
# N = 20000; Trials = c(1:17)
# N = 20000; Trials = c(18:34)
# N = 20000; Trials = c(35:50)
# N = 50000; Trials = c(1:13)
# N = 50000; Trials = c(14:25)
# N = 50000; Trials = c(26:38)
# N = 50000; Trials = c(39:50)
# N = 100000; Trials = c(1:10)
# N = 100000; Trials = c(11:20)
# N = 100000; Trials = c(21:30)
# N = 100000; Trials = c(31:40)
# N = 100000; Trials = c(41:50)






##### Data Generation
set.seed(a+b*seed)
source("subcodes/1D Data Generation.R")
var_Z_true = sig2_true / (1-gamma^2) # Assume to be known.
source("subcodes/1D_Realizable Objective Functions.R")


###### Minimization

theta_initial = c(A_candidate=1, p_candidate=0.1)
lower_bounds = c(0.1,0.01)
upper_bounds = c(10000, 0.99)


#### EBRM-singlestep

res <- optim(par=theta_initial, fn=Energy_singlestep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
(EBRM_singlestep_estimator = res$par)


#### EBRM-multistep

M_size = C1 * N

disparity_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(disparity_table) = paste0("m=",mstep_vec)
rownames(disparity_table) = paste0("trial", Trials)

A_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(A_table) = paste0("m=",mstep_vec)
rownames(A_table) = paste0("trial", Trials)

rho_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(rho_table) = paste0("m=",mstep_vec)
rownames(rho_table) = paste0("trial", Trials)


for(mstep_index in 1:length(mstep_vec)){
  
  # mstep_index=1
  
  mstep = mstep_vec[mstep_index]
  
  for(trial_index in 1:length(Trials)){
    
    tictoc::tic()
    
    # trial_index=25

    set.seed(a/10 * Trials[trial_index] + b * seed)
    source("subcodes/1D Resample Trajectories.R")

    res <- optim(par=theta_initial, fn=Energy_multistep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, maxit=200), lower=lower_bounds, upper=upper_bounds)
    EBRM_multistep_estimator = res$par
    
    disparity_table[trial_index, mstep_index]=Disparity_thetas(EBRM_multistep_estimator, EBRM_singlestep_estimator)
    A_table[trial_index, mstep_index] = EBRM_multistep_estimator[1]
    rho_table[trial_index, mstep_index] = EBRM_multistep_estimator[2]
    
    tictoc::toc()
    
  }
  
  print(paste0("--------------------------step=",mstep," completed."))
  
}


# disparity_table
# apply(disparity_table, 2, mean)
# apply(disparity_table, 2, sd)


### Plot and save the result.

# par(mfrow=c(1,1))
# 
# matplot(mstep_vec, t(disparity_table), xlab = "step level (m)", ylab="Disparity", main="")
# title(paste0("Disparity from 1-step: Realizable & sig2=",sig2_true ))
# lines(mstep_vec, apply(disparity_table, 2, mean))



filename=paste0("EBRM_realizable_disparity_var",sig2_true, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(disparity_table, file=filename)

filename=paste0("EBRM_realizable_Avalue_var",sig2_true, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(A_table, file=filename)

filename=paste0("EBRM_realizable_rhovalue_var",sig2_true, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(rho_table, file=filename)






