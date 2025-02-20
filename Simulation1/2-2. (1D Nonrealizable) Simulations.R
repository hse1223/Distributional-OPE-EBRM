rm(list=ls())
# dev.off()

##### Parameter Setting

a=54325347; b=4326463
seedvec=c(1:100)

### Environment
mu_last = 0
gridno = 100


### Setting

# gamma = 0.5; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9 # setting2 - small gamma
# # N = 2000; mstep=1; C1=1
# # N = 3000; mstep=1; C1=1
# # N = 5000; mstep=1; C1=1
# # N = 10000; mstep=2; C1=1
# criterion_T=10; criterion_N=3000; minimum_T=10; p=0.7 # FLE

gamma = 0.99; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9 # setting1 - large gamma
# N = 2000; mstep=120; C1=1
# N = 3000; mstep=180; C1=1
# N = 5000; mstep=220; C1=1
N = 10000; mstep=240; C1=1
criterion_T=25; criterion_N=2000; minimum_T=15; p=10 # FLE

##### Simulations

Inaccuracy_EBRMmulti = Inaccuracy_FLE = Time_EBRMmulti = Time_FLE = rep(NA, length(seedvec))

Parameter_EBRMmulti = Parameter_FLE = matrix(NA, length(seedvec), 4)
rownames(Parameter_EBRMmulti) = rownames(Parameter_FLE) = paste0("seed",seedvec)
colnames(Parameter_EBRMmulti) = colnames(Parameter_FLE) = c("beta0_L", "beta0_R", "beta1", "sig2")

#### Tuning Parameters - EBRM-mstep

M_size = C1 * N

#### Tuning Parameters - FLE

C_value = gamma^((1-1/(2*p)) * criterion_T)  * (criterion_N / log(criterion_N))^(1/(2*p))
partition = log(C_value * (log(N)/N)^(1/(2*p))) / log(gamma^(1-1/(2*p)))
if(abs(partition - criterion_T) < 1e-10){
  partition = criterion_T
}
partition = max(partition, minimum_T)
(partition = floor(partition))
(subdata_size = N %/% partition)



#### Implementations

var_Z_true = sig2_true / (1-gamma^2) # Assume to be known.

for(seed_index in 1:length(seedvec)){
  
  # seed_index=1
  
  set.seed(a+b*seedvec[seed_index])
  
  ### Data Generation & Initial Values
  
  source("subcodes/1D Data Generation.R")
  source("subcodes/1D_Nonrealizable Objective Functions.R")
  
  theta_initial = c("beta0_L" = 0, "beta0_R" = 0, "beta1" = 0, "sig2" = 1)
  lower_bounds = c(-100000,-100000,-200000, 1)
  upper_bounds = c(200000, 200000, 100000, 100000000)
  
  
  ### EBRM-multistep
  
  tictoc::tic()
  
  if (mstep==1){
    res <- optim(par=theta_initial, fn=Energy_singlestep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
  } else {
    source("subcodes/1D Resample Trajectories.R")
    res <- optim(par=theta_initial, fn=Energy_multistep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
  }
  EBRM_multistep_estimator = res$par
  
  time_measurement = tictoc::toc()
  time_measurement = time_measurement$callback_msg
  time_measurement = as.numeric(stringr::str_replace(string=time_measurement, pattern=" sec elapsed", replacement = ""))
  Time_EBRMmulti[seed_index] = time_measurement
  Inaccuracy_EBRMmulti[seed_index] = Inaccuracy(EBRM_multistep_estimator)
  Parameter_EBRMmulti[seed_index,] = EBRM_multistep_estimator
  
  
  ### FLE
  
  tictoc::tic()
  
  source("subcodes/1D_Nonrealizable FLE Iterations.R")
  FLE_estimator = theta_prev
  
  time_measurement = tictoc::toc()
  time_measurement = time_measurement$callback_msg
  time_measurement = as.numeric(stringr::str_replace(string=time_measurement, pattern=" sec elapsed", replacement = ""))
  Time_FLE[seed_index] = time_measurement
  Inaccuracy_FLE[seed_index] = Inaccuracy(FLE_estimator)
  Parameter_FLE[seed_index,] = FLE_estimator
  
  print(paste0("------------------ seed=",seed_index, " Completed."))
}



##### Save Results

### EBRM_multi

rdataname = paste0("EBRMmulti_gamma",gamma*100, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".rdata")
save(Parameter_EBRMmulti, file=rdataname)
Inaccuracy_Time_EBRMmulti = cbind(Inaccuracy = Inaccuracy_EBRMmulti, Seconds = Time_EBRMmulti)
rownames(Inaccuracy_Time_EBRMmulti) = paste0("seed=", seedvec)
filename_EBRMmulti = paste0("EBRMmulti_gamma",gamma*100, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".txt")
write.table(x=Inaccuracy_Time_EBRMmulti, file=filename_EBRMmulti)


### FLE

rdataname = paste0("FLE_gamma",gamma*100, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".rdata")
save(Parameter_FLE, file=rdataname)
Inaccuracy_Time_FLE = cbind(Inaccuracy = Inaccuracy_FLE, Seconds = Time_FLE)
rownames(Inaccuracy_Time_FLE) = paste0("seed=", seedvec)
filename_FLE = paste0("FLE_gamma",gamma*100, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".txt")
write.table(x=Inaccuracy_Time_FLE, file=filename_FLE)


##### Brief Summary

round(Inaccuracy_EBRMmulti, 5)
round(Inaccuracy_FLE,5)









