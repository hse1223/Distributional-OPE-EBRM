Inaccuracy_Parameter_mix_Wasserstein <- function(theta_vec){
  
  # theta_vec = Parameter_EBRMmulti[sim_ind,]
  
  muZ_left = A_true*p_true^(svec-1) * (1-(gamma*p_true)^(S_size-svec+2)) / (1-gamma*p_true) + gamma^(S_size-svec+2) / (1-gamma) * mu_last # E(Z(s,L))
  muZ_right = A_true*p_true^(svec+1) * (1-(gamma*p_true)^(S_size-svec)) / (1-gamma*p_true) + gamma^(S_size-svec) / (1-gamma) * mu_last # E(Z(s,R))
  muZ_left[1] = gamma * muZ_right[1] + muR_left[1] # exception for (s,a)=(1,L)
  
  beta0_L = theta_vec[1]
  beta0_R = theta_vec[2]
  beta1 = theta_vec[3]
  sig2 = theta_vec[4]
  
  muZ_left_candidate_vec = beta0_L + beta1 * svec
  muZ_right_candidate_vec = beta0_R + beta1 * svec
  var_Z_candidate = sig2 / (1-gamma^2)
  
  muZ_candidate_mat = rbind(muZ_left_candidate_vec, muZ_right_candidate_vec)
  rownames(muZ_candidate_mat) = c("left", "right")
  colnames(muZ_candidate_mat) = paste0("S",1:S_size)
  
  ## How many (s,a) to collect
  
  left_samplesize_candidate = rbinom(n=1, size=Wasserstein_samples, prob=1/2)
  right_samplesize_candidate = Wasserstein_samples - left_samplesize_candidate
  left_states_candidate=sample(x=1:S_size, size=left_samplesize_candidate, replace = T)
  right_states_candidate=sample(x=1:S_size, size=right_samplesize_candidate, replace = T)
  
  samplesize_candidate_sa = matrix(0, 2, S_size)
  rownames(samplesize_candidate_sa) = c("left", "right")
  colnames(samplesize_candidate_sa) = paste0("S",1:S_size)
  for(s_index in 1:S_size){
    # s_index=1
    samplesize_candidate_sa[1, s_index]=sum(left_states_candidate==s_index)
    samplesize_candidate_sa[2, s_index]=sum(right_states_candidate==s_index)
  }
  
  # rm(left_samplesize_candidate, right_samplesize_candidate, left_states_candidate, right_states_candidate)
  
  left_samplesize_true = rbinom(n=1, size=Wasserstein_samples, prob=1/2)
  right_samplesize_true = Wasserstein_samples - left_samplesize_true
  left_states_true=sample(x=1:S_size, size=left_samplesize_true, replace = T)
  right_states_true=sample(x=1:S_size, size=right_samplesize_true, replace = T)
  
  samplesize_true_sa = matrix(0, 2, S_size)
  rownames(samplesize_true_sa) = c("left", "right")
  colnames(samplesize_true_sa) = paste0("S",1:S_size)
  for(s_index in 1:S_size){
    # s_index=1
    samplesize_true_sa[1, s_index]=sum(left_states_true==s_index)
    samplesize_true_sa[2, s_index]=sum(right_states_true==s_index)
  }
  
  # rm(left_samplesize_true, right_samplesize_true, left_states_true, right_states_true)

  
  ## Sample Z
  
  Wassersamples_candidate = Wassersamples_true = NULL
  
  for(state_index in 1:S_size){
    
    # state_index=1
    
    left_samples_candidate=rnorm(samplesize_candidate_sa[1, state_index], mean = muZ_left_candidate_vec[state_index], sd=sqrt(var_Z_candidate))    
    right_samples_candidate=rnorm(samplesize_candidate_sa[2, state_index], mean = muZ_right_candidate_vec[state_index], sd=sqrt(var_Z_candidate))    
    Wassersamples_candidate = c(Wassersamples_candidate, left_samples_candidate, right_samples_candidate)
    
    left_samples_true=rnorm(samplesize_true_sa[1, state_index], mean = muZ_left[state_index], sd=sqrt(var_Z_true))    
    right_samples_true=rnorm(samplesize_true_sa[2, state_index], mean = muZ_right[state_index], sd=sqrt(var_Z_true))    
    Wassersamples_true = c(Wassersamples_true, left_samples_true, right_samples_true)
  }
  
  ## Calculate Wasserstein-1 metric
  
  Wasser_value = transport::wasserstein1d(Wassersamples_candidate, Wassersamples_true)
  return(Wasser_value)  
}




Inaccuracy_QRTD_mix_Wasserstein <- function(Qmats){
  
  # Qmats = Qmats_list[[sim_ind]]
  
  muZ_left = A_true*p_true^(svec-1) * (1-(gamma*p_true)^(S_size-svec+2)) / (1-gamma*p_true) + gamma^(S_size-svec+2) / (1-gamma) * mu_last # E(Z(s,L))
  muZ_right = A_true*p_true^(svec+1) * (1-(gamma*p_true)^(S_size-svec)) / (1-gamma*p_true) + gamma^(S_size-svec) / (1-gamma) * mu_last # E(Z(s,R))
  muZ_left[1] = gamma * muZ_right[1] + muR_left[1] # exception for (s,a)=(1,L)
  
  
  ## How many (s,a) to collect
  
  left_samplesize_true = rbinom(n=1, size=Wasserstein_samples, prob=1/2)
  right_samplesize_true = Wasserstein_samples - left_samplesize_true
  left_states_true=sample(x=1:S_size, size=left_samplesize_true, replace = T)
  right_states_true=sample(x=1:S_size, size=right_samplesize_true, replace = T)
  
  samplesize_true_sa = matrix(0, 2, S_size)
  rownames(samplesize_true_sa) = c("left", "right")
  colnames(samplesize_true_sa) = paste0("S",1:S_size)
  for(s_index in 1:S_size){
    # s_index=1
    samplesize_true_sa[1, s_index]=sum(left_states_true==s_index)
    samplesize_true_sa[2, s_index]=sum(right_states_true==s_index)
  }
  
  # rm(left_samplesize_true, right_samplesize_true, left_states_true, right_states_true)
  
  
  ## Sample Z
  
  Qmat_samples = c(Qmats$left, Qmats$right)
  Wassersamples_candidate = sample(x=Qmat_samples, size=Wasserstein_samples, replace = T)
  
  Wassersamples_true = NULL
  for(state_index in 1:S_size){
    left_samples_true=rnorm(samplesize_true_sa[1, state_index], mean = muZ_left[state_index], sd=sqrt(var_Z_true))    
    right_samples_true=rnorm(samplesize_true_sa[2, state_index], mean = muZ_right[state_index], sd=sqrt(var_Z_true))    
    Wassersamples_true = c(Wassersamples_true, left_samples_true, right_samples_true)
  }
  
  ## Calculate Wasserstein-1 metric
  
  Wasser_value = transport::wasserstein1d(Wassersamples_candidate, Wassersamples_true)
  return(Wasser_value)  
}





















