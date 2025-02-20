rm(list=ls())
# dev.off()

##### Parameter Setting

a=54325347; b=4326463

### Environment
gamma = 0.99
mu_last = 0
gridno = 100


### Setting

S_size=30; A_true = 100; sig2_true = 20; p_true=0.9; N = 20000; seed=1 # setting1 - small variance
criterion_T=25; criterion_N=2000; minimum_T=15; p=10 # FLE
alpha = 5; tau <- 1:99/100; update_per_sample = length(tau) # QRTD


# S_size=30; A_true = 100; sig2_true = 5000; p_true=0.9; N = 5000; seed=1 # setting2 - big variance
# criterion_T=25; criterion_N=20000; minimum_T=15; p=10 # FLE
# alpha = 2; tau <- 1:99/100; update_per_sample = length(tau) # QRTD


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


#### FLE

## Tuning Parameters

C_value = gamma^((1-1/(2*p)) * criterion_T)  * (criterion_N / log(criterion_N))^(1/(2*p))
partition = log(C_value * (log(N)/N)^(1/(2*p))) / log(gamma^(1-1/(2*p)))
if(abs(partition - criterion_T) < 1e-10){
  partition = criterion_T
}
partition = max(partition, minimum_T)
(partition = floor(partition))
(subdata_size = N %/% partition)

## Iterations

source("subcodes/1D_Realizable FLE Iterations.R")
(FLE_estimator = theta_prev)



#### QRTD

source("subcodes/1D_QRTD iterations.R")

par(mfrow=c(1,2), oma=c(0,0,2,0))
matplot(1:S_size, t(Qmats$left), main = "Q(s,left)")
matplot(1:S_size, t(Qmats$right), main = "Q(s,right)")
mtext(paste0("time = ", time), outer=T, cex=2, line=-1)




#### Estimators

EBRM_singlestep_estimator
FLE_estimator
Qmats

Inaccuracy(EBRM_singlestep_estimator)
Inaccuracy(FLE_estimator)



