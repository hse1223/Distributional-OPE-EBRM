rm(list=ls())
# dev.off()

##### Parameter Setting

a=54325347; b=4326463
seedvec=c(1:100)

### Environment

mu_last = 0
gridno = 100


### Setting

gamma = 0.99; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; N = 3000 # setting1 - small variance
criterion_T=25; criterion_N=2000; minimum_T=15; p=10 # FLE
alpha = 5; tau <- 1:99/100; update_per_sample = length(tau) # QRTD

# gamma = 0.99; S_size=30; A_true = 100; sig2_true = 5000; p_true=0.9; N = 100000 # setting2 - big variance
# criterion_T=25; criterion_N=20000; minimum_T=15; p=10 # FLE
# alpha = 2; tau <- 1:99/100; update_per_sample = length(tau) # QRTD

# gamma = 0.5; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; N=3000 # small gamma => Only leave QRTD codes.
# alpha = 3; tau <- 1:99/100; update_per_sample = length(tau)  # QRTD



##### Simulations

# Inaccuracy_EBRMsingle = Inaccuracy_FLE = Time_EBRMsingle = Time_FLE = rep(NA, length(seedvec))
# 
# Parameter_EBRMsingle = Parameter_FLE = matrix(NA, length(seedvec), 2)
# rownames(Parameter_EBRMsingle) = rownames(Parameter_FLE) = paste0("seed",seedvec)
# colnames(Parameter_EBRMsingle) = colnames(Parameter_FLE) = c("A", "p")


Qmats_list = vector("list", length(seedvec))
names(Qmats_list) = paste0("seed=",seedvec)
Time_QRTD = rep(NA, length(seedvec))


# #### Tuning Parameters - FLE
# 
# C_value = gamma^((1-1/(2*p)) * criterion_T)  * (criterion_N / log(criterion_N))^(1/(2*p))
# partition = log(C_value * (log(N)/N)^(1/(2*p))) / log(gamma^(1-1/(2*p)))
# if(abs(partition - criterion_T) < 1e-10){
#   partition = criterion_T
# }
# partition = max(partition, minimum_T)
# (partition = floor(partition))
# (subdata_size = N %/% partition)


#### Implementations

for(seed_index in 1:length(seedvec)){
  
  # seed_index=1
  
  set.seed(a+b*seedvec[seed_index])
  
  ### Data Generation & Initial Values
  
  source("subcodes/1D Data Generation.R")
  var_Z_true = sig2_true / (1-gamma^2) # Assume to be known.
  source("subcodes/1D_Realizable Objective Functions.R")
  
  theta_initial = c(A_candidate=1, p_candidate=0.1)
  lower_bounds = c(0.1,0.01)
  upper_bounds = c(10000, 0.99)
  
  
  # ### EBRM-singlestep
  # 
  # tictoc::tic()
  # 
  # res <- optim(par=theta_initial, fn=Energy_singlestep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
  # EBRM_singlestep_estimator = res$par
  # 
  # time_measurement = tictoc::toc()
  # time_measurement = time_measurement$callback_msg
  # time_measurement = as.numeric(stringr::str_replace(string=time_measurement, pattern=" sec elapsed", replacement = ""))
  # Time_EBRMsingle[seed_index] = time_measurement
  # Inaccuracy_EBRMsingle[seed_index] = Inaccuracy(EBRM_singlestep_estimator)
  # Parameter_EBRMsingle[seed_index,] = EBRM_singlestep_estimator
  # 
  # 
  # ### FLE
  # 
  # tictoc::tic()
  # 
  # source("subcodes/1D_Realizable FLE Iterations.R")
  # FLE_estimator = theta_prev
  # 
  # time_measurement = tictoc::toc()
  # time_measurement = time_measurement$callback_msg
  # time_measurement = as.numeric(stringr::str_replace(string=time_measurement, pattern=" sec elapsed", replacement = ""))
  # Time_FLE[seed_index] = time_measurement
  # Inaccuracy_FLE[seed_index] = Inaccuracy(FLE_estimator)
  # Parameter_FLE[seed_index,] = FLE_estimator
  
  
  ### QRTD
  
  tictoc::tic()
  source("subcodes/1D_QRTD iterations.R")
  time_measurement = tictoc::toc()
  time_measurement = time_measurement$callback_msg
  time_measurement = as.numeric(stringr::str_replace(string=time_measurement, pattern=" sec elapsed", replacement = ""))
  Time_QRTD[seed_index] = time_measurement
  Qmats_list[[seed_index]] = Qmats
  

  print(paste0("------------------ seed=",seed_index, " Completed."))
}


##### Save Results

# ### EBRM_single
# 
# rdataname = paste0("EBRMsingle_var",sig2_true, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".rdata")
# save(Parameter_EBRMsingle, file=rdataname)
# Inaccuracy_Time_EBRMsingle = cbind(Inaccuracy = Inaccuracy_EBRMsingle, Seconds = Time_EBRMsingle)
# rownames(Inaccuracy_Time_EBRMsingle) = paste0("seed=", seedvec)
# filename_EBRMsingle = paste0("EBRMsingle_var",sig2_true, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".txt")
# write.table(x=Inaccuracy_Time_EBRMsingle, file=filename_EBRMsingle)
# 
# 
# ### FLE
# 
# rdataname = paste0("FLE_var",sig2_true, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".rdata")
# save(Parameter_FLE, file=rdataname)
# Inaccuracy_Time_FLE = cbind(Inaccuracy = Inaccuracy_FLE, Seconds = Time_FLE)
# rownames(Inaccuracy_Time_FLE) = paste0("seed=", seedvec)
# filename_FLE = paste0("FLE_var",sig2_true, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".txt")
# write.table(x=Inaccuracy_Time_FLE, file=filename_FLE)


### QRTD

rdataname = paste0("QRTD_var",sig2_true, "N",N, "_seed",seedvec[1],"to",seedvec[length(seedvec)], ".rdata")
save(Qmats_list, file=rdataname)


# ##### Brief Summary
# 
# round(Inaccuracy_EBRMsingle, 5)
# round(Inaccuracy_FLE,5)



















