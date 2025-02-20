rm(list=ls())
# dev.off()

##### Parameter Setting

a=54325347; b=4326463

### Environment
mu_last = 0
gridno = 100 


### Setting

gamma = 0.99; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; N = 2000; seed=3 # setting2 - original
C1=1 # Muti-step EBRM
mstep_vec = c(1:13)*20 # nor recommended if gamma^m < 0.05 => numerical problems.

N = 2000; Trials = c(1:25)
# N = 2000; Trials = c(26:50)
# N = 3000; Trials = c(1:25)
# N = 3000; Trials = c(26:50)
# N = 5000; Trials = c(1:25)
# N = 5000; Trials = c(26:50)
# N = 10000; Trials = c(1:25)
# N = 10000; Trials = c(26:50)


# gamma = 0.5; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; N = 20000; seed=6 # setting2 - original
# C1=1 # Muti-step EBRM
# mstep_vec = c(2:5)
# N = 2000; Trials = c(1:50)
# N = 3000; Trials = c(1:50)
# N = 5000; Trials = c(1:50)
# N = 10000; Trials = c(1:50)


##### Data Generation

set.seed(a+b*seed)
source("subcodes/1D Data Generation.R")
source("subcodes/1D_Nonrealizable Objective Functions.R")



###### Minimization

theta_initial = c("beta0_L" = 0, "beta0_R" = 0, "beta1" = 0, "sig2" = 1) 
lower_bounds = c(-10000,-10000,-20000, 1)
upper_bounds = c(10000, 10000, 20000, 1000000)


#### EBRM-singlestep

res <- optim(par=theta_initial, fn=Energy_singlestep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
(EBRM_singlestep_estimator = res$par)


#### EBRM-multistep

M_size = C1 * N

disparity_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(disparity_table) = paste0("m=",mstep_vec)
rownames(disparity_table) = paste0("trial", Trials)


beta0L_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(beta0L_table) = paste0("m=",mstep_vec)
rownames(beta0L_table) = paste0("trial", Trials)

beta0R_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(beta0R_table) = paste0("m=",mstep_vec)
rownames(beta0R_table) = paste0("trial", Trials)

beta1_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(beta1_table) = paste0("m=",mstep_vec)
rownames(beta1_table) = paste0("trial", Trials)

sig2_table = matrix(NA, length(Trials), length(mstep_vec))
colnames(sig2_table) = paste0("m=",mstep_vec)
rownames(sig2_table) = paste0("trial", Trials)


for(mstep_index in 1:length(mstep_vec)){
  
  mstep = mstep_vec[mstep_index]
  
  # mstep=250
  
  for(trial_index in 1:length(Trials)){
    
    tictoc::tic()
    
    set.seed(a/10 * Trials[trial_index] + b * seed)
    
    source("subcodes/1D Resample Trajectories.R")
    res <- optim(par=theta_initial, fn=Energy_multistep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, maxit=200), lower=lower_bounds, upper=upper_bounds)
    EBRM_multistep_estimator = res$par
    
    disparity_table[trial_index, mstep_index]=Disparity_thetas(EBRM_multistep_estimator, EBRM_singlestep_estimator)
    beta0L_table[trial_index, mstep_index] = EBRM_multistep_estimator[1]
    beta0R_table[trial_index, mstep_index] = EBRM_multistep_estimator[2]
    beta1_table[trial_index, mstep_index] = EBRM_multistep_estimator[3]
    sig2_table[trial_index, mstep_index] = EBRM_multistep_estimator[4]

    tictoc::toc()
    
  }
  
  print(paste0("--------------------------step=",mstep," completed."))
  
}

# disparity_table
# 
# apply(disparity_table, 2, mean)
# apply(disparity_table, 2, sd)
# 
# matplot(mstep_vec, t(disparity_table), xlab = "step level (m)", ylab="Disparity", main="")
# title(paste0("Disparity from 1-step: Non-realizable & sig2=",sig2_true ))
# lines(mstep_vec, apply(disparity_table, 2, mean))


### Plot and save the result.

# par(mfrow=c(1,1))
# 
# matplot(mstep_vec, t(disparity_table), xlab = "step level (m)", ylab="Disparity", main="")
# title(paste0("Disparity from 1-step: Nonrealizable & gamma=",gamma ))
# lines(mstep_vec, apply(disparity_table, 2, mean))

filename=paste0("EBRM_nonrealizable_disparity_gamma",gamma*100, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(disparity_table, file=filename)

filename=paste0("EBRM_nonrealizable_beta0Lvalue_gamma",gamma*100, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(beta0L_table, file=filename)

filename=paste0("EBRM_nonrealizable_beta0Rvalue_gamma",gamma*100, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(beta0R_table, file=filename)

filename=paste0("EBRM_nonrealizable_beta1value_gamma",gamma*100, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(beta1_table, file=filename)

filename=paste0("EBRM_nonrealizable_sig2value_gamma",gamma*100, "N",N,"Trial",Trials[1],"to",Trials[length(Trials)],".txt")
write.table(sig2_table, file=filename)




