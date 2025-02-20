##### Realizable

rm(list=ls())

a=54325347; b=4326463

### Choose

sig2_true=20; N_vec = c(500, 1000, 2000, 5000, 10000, 20000)
# sig2_true=5000; N_vec = c(2000, 5000, 10000, 20000, 50000, 1e+05)
gridno=100
Wasserstein_samples = 1000000
# Wasserstein_samples = 10000


### Settings

S_size=30; A_true = 100; p_true=0.9; gamma=0.99
svec=1:S_size; mu_last=0
muR_left = A_true * p_true^(svec - 1) # E(R(s,L))
muR_right = A_true * p_true^(svec + 1); muR_right[S_size] = mu_last # E(R(s,R))
var_Z_true = sig2_true / (1-gamma^2) # Assume to be known.


### Inaccuracy

source("subcodes/mixture_inaccuracy/1D_mixture inaccuracy - realizable.R")

EBRM_filenames = paste0("results/simulations/realizable/var",sig2_true,"/EBRMsingle_var",sig2_true,"N",N_vec,"_seed1to100.rdata")
FLE_filenames = paste0("results/simulations/realizable/var",sig2_true,"/FLE_var",sig2_true,"N",N_vec,"_seed1to100.rdata")
QRTD_filenames = paste0("results/simulations/realizable/var",sig2_true,"/QRTD_var",sig2_true,"N",N_vec,"_seed1to100.rdata")

EBRM_Wassermix_Inaccuracy = FLE_Wassermix_Inaccuracy = QRTD_Wassermix_Inaccuracy = matrix(NA, 100, length(N_vec))
rownames(EBRM_Wassermix_Inaccuracy) = rownames(FLE_Wassermix_Inaccuracy) = rownames(QRTD_Wassermix_Inaccuracy) = paste0("seed=",1:100)
colnames(EBRM_Wassermix_Inaccuracy) = colnames(FLE_Wassermix_Inaccuracy) = colnames(QRTD_Wassermix_Inaccuracy) = paste0("N=", N_vec)


for(i in 1:length(EBRM_filenames)){
  
  # i=1
  
  load(EBRM_filenames[i])
  load(FLE_filenames[i])
  load(QRTD_filenames[i])
  
  for(sim_ind in 1:100){
    
    # sim_ind=1
    
    set.seed(a+b*sim_ind*1)
    EBRM_Wassermix_Inaccuracy[sim_ind, i] = Inaccuracy_Parameter_mix_Wasserstein(Parameter_EBRMsingle[sim_ind,])
    
    set.seed(a+b*sim_ind*2)
    FLE_Wassermix_Inaccuracy[sim_ind, i] = Inaccuracy_Parameter_mix_Wasserstein(Parameter_FLE[sim_ind,])
    
    set.seed(a+b*sim_ind*3)
    QRTD_Wassermix_Inaccuracy[sim_ind, i] = Inaccuracy_QRTD_mix_Wasserstein(Qmats_list[[sim_ind]])
    
    print(paste0("seed=",sim_ind, " completed."))
    
  }
  
  print(paste0("----------N=",N_vec[i], " completed."))
}

round(apply(EBRM_Wassermix_Inaccuracy, 2, mean),3)
round(apply(EBRM_Wassermix_Inaccuracy, 2, sd),3)

round(apply(FLE_Wassermix_Inaccuracy, 2, mean),3)
round(apply(FLE_Wassermix_Inaccuracy, 2, sd),3)

round(apply(QRTD_Wassermix_Inaccuracy, 2, mean),3)
round(apply(QRTD_Wassermix_Inaccuracy, 2, sd),3)





##### Non-realizable 

rm(list=ls())

a=54325347; b=4326463

### Choose

# gamma=0.50; N_vec = c(2000, 3000, 5000, 10000)
gamma=0.99; N_vec = c(2000, 3000, 5000, 10000)

Wasserstein_samples = 1000000
# Wasserstein_samples = 10000


### Settings

S_size=30; A_true = 100; p_true=0.9; sig2_true = 20
svec=1:S_size; mu_last=0
muR_left = A_true * p_true^(svec - 1) # E(R(s,L))
muR_right = A_true * p_true^(svec + 1); muR_right[S_size] = mu_last # E(R(s,R))
gridno=100
var_Z_true = sig2_true / (1-gamma^2) # Assume to be known.


### Inaccuracy

source("subcodes/mixture_inaccuracy/1D_mixture inaccuracy - nonrealizable.R")

EBRM_filenames = paste0("results/simulations/nonrealizable/gamma",gamma*100,"/EBRMmulti_gamma", gamma*100,"N",N_vec,"_seed1to100.rdata")
FLE_filenames = paste0("results/simulations/nonrealizable/gamma",gamma*100,"/FLE_gamma", gamma*100,"N",N_vec,"_seed1to100.rdata")

if(gamma==0.99){
  QRTD_filenames = paste0("results/simulations/realizable/var",sig2_true,"/QRTD_var",sig2_true,"N",N_vec,"_seed1to100.rdata")
} else if (gamma==0.50){
  QRTD_filenames = paste0("results/simulations/nonrealizable/gamma",gamma*100,"/QRTD_var",sig2_true,"N",N_vec,"_seed1to100.rdata")
} else{
  stop("no such gamma.")
}

EBRM_Wassermix_Inaccuracy = FLE_Wassermix_Inaccuracy = QRTD_Wassermix_Inaccuracy = matrix(NA, 100, length(N_vec))
rownames(EBRM_Wassermix_Inaccuracy) = rownames(FLE_Wassermix_Inaccuracy) = rownames(QRTD_Wassermix_Inaccuracy) = paste0("seed=",1:100)
colnames(EBRM_Wassermix_Inaccuracy) = colnames(FLE_Wassermix_Inaccuracy) = colnames(QRTD_Wassermix_Inaccuracy) = paste0("N=", N_vec)


for(i in 1:length(EBRM_filenames)){
  
  # i=1
  
  load(EBRM_filenames[i])
  load(FLE_filenames[i])
  load(QRTD_filenames[i])
  
  
  for(sim_ind in 1:100){
    
    # sim_ind=1
    
    set.seed(a+b*sim_ind*1)
    EBRM_Wassermix_Inaccuracy[sim_ind, i] = Inaccuracy_Parameter_mix_Wasserstein(Parameter_EBRMmulti[sim_ind,])
    
    set.seed(a+b*sim_ind*2)
    FLE_Wassermix_Inaccuracy[sim_ind, i] = Inaccuracy_Parameter_mix_Wasserstein(Parameter_FLE[sim_ind,])
    
    # set.seed(a+b*sim_ind*3)
    # QRTD_Wassermix_Inaccuracy[sim_ind, i] = Inaccuracy_QRTD_mix_Wasserstein(Qmats_list[[sim_ind]])
    
    print(paste0("seed=",sim_ind, " completed."))
  }
  
  print(paste0("----------N=",N_vec[i], " completed."))
}

round(apply(EBRM_Wassermix_Inaccuracy, 2, mean),3)
round(apply(EBRM_Wassermix_Inaccuracy, 2, sd),3)

round(apply(FLE_Wassermix_Inaccuracy, 2, mean),3)
round(apply(FLE_Wassermix_Inaccuracy, 2, sd),3)

round(apply(QRTD_Wassermix_Inaccuracy, 2, mean),3)
round(apply(QRTD_Wassermix_Inaccuracy, 2, sd),3)



