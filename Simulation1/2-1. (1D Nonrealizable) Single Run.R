rm(list=ls())
# dev.off()

##### Parameter Setting

a=54325347; b=4326463

### Environment
mu_last = 0
gridno = 100


### Setting

# gamma = 0.99; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; seed=1 # setting1 - small variance
# N = 3000; mstep=100; C1=1
# criterion_T=25; criterion_N=2000; minimum_T=15; p=10 # FLE

gamma = 0.5; S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; seed=2 # setting1 - small variance
N = 10000; mstep=2; C1=1
criterion_T=10; criterion_N=3000; minimum_T=10; p=0.7 # FLE

##### Data Generation
set.seed(a+b*seed)
source("subcodes/1D Data Generation.R")
source("subcodes/1D_Nonrealizable Objective Functions.R")


###### Minimization

theta_initial = c("beta0_L" = 0, "beta0_R" = 0, "beta1" = 0, "sig2" = 1)
# lower_bounds = c(-10,-10,-200, 1)
# upper_bounds = c(2000, 2000, 100, 1000)
lower_bounds = c(-100000,-100000,-200000, 1)
upper_bounds = c(200000, 200000, 100000, 100000000)


var_Z_true = sig2_true / (1-gamma^2) # Assume to be known.
res <- optim(par=theta_initial, fn=Inaccuracy, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
theta_pseudo = res$par


#### EBRM-multistep

M_size = C1 * N

if (mstep==1){
  res <- optim(par=theta_initial, fn=Energy_singlestep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
} else {
  source("subcodes/1D Resample Trajectories.R")
  res <- optim(par=theta_initial, fn=Energy_multistep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
}
(EBRM_multistep_estimator = res$par)




#### FLE

## Tuning Parameters (same as Realizable)

C_value = gamma^((1-1/(2*p)) * criterion_T)  * (criterion_N / log(criterion_N))^(1/(2*p))
partition = log(C_value * (log(N)/N)^(1/(2*p))) / log(gamma^(1-1/(2*p)))
if(abs(partition - criterion_T) < 1e-10){
  partition = criterion_T
}
partition = max(partition, minimum_T)
(partition = floor(partition))
(subdata_size = N %/% partition)
# partition=100; subdata_size=10000


## Iterations

source("subcodes/1D_Nonrealizable FLE Iterations.R")
(FLE_estimator = theta_prev)

parameter_matrix


#### Inaccuracy

Inaccuracy(theta_pseudo)
Inaccuracy(EBRM_multistep_estimator)
Inaccuracy(FLE_estimator)






