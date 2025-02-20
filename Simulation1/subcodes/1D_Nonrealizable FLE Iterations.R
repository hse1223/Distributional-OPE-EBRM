## Integrate the Data List into a single dataframe.

RSprime_df = vector("list", S_size)
for(sindex in 1:S_size){
  # sindex=1
  SL_data = data.frame(S=sindex, A="left",RSprime[[sindex]]$left)
  SR_data = data.frame(S=sindex, A="right",RSprime[[sindex]]$right)
  RSprime_df[[sindex]] = rbind(SL_data, SR_data)
}
RSprime_df = do.call(rbind, RSprime_df)


if(sum(is.na(RSprime_df$Reward)) > 0){
  RSprime_df = RSprime_df[-which(is.na(RSprime_df$Reward)),]
}

theta_prev = theta_initial
index_mixed = sample(x=1:N, size=N, replace=FALSE) 
# index_mixed = 1:N

parameter_matrix = matrix(NA, 4, partition+1)
parameter_matrix[,1] = theta_prev
rownames(parameter_matrix) = c("beta0_L", "beta0_R", "beta1", "sig2")


## Iterations

for(iteration in 1:partition){
  
  # iteration = 1
  
  if (iteration < partition){
    index_sub = 1:subdata_size + subdata_size * (iteration-1)
  } else {
    index_sub = (subdata_size * (iteration-1) + 1):N # use up all the remainders.
  }
  
  index_chosen=index_mixed[index_sub]
  index_chosen = sort(index_chosen)
  RSprime_df_sub = RSprime_df[index_chosen,]
  
  
  beta0_L_prev = theta_prev[1]
  beta0_R_prev = theta_prev[2]
  beta1_prev = theta_prev[3]
  sig2_prev = theta_prev[4]
  
  muZ_left_prev_vec = beta0_L_prev + beta1_prev * svec
  muZ_right_prev_vec = beta0_R_prev + beta1_prev * svec
  muZ_prev_mat = rbind(muZ_left_prev_vec, muZ_right_prev_vec)
  rownames(muZ_prev_mat) = c("left", "right")
  var_Z_prev = sig2_prev / (1-gamma^2)
  
  Z_samples = rep(NA, nrow(RSprime_df_sub))
  mu_sub = rep(NA, nrow(RSprime_df_sub))
  nsa_sub = c()

    
  for(sindex in 1:S_size){
    
    # sindex=1
    
    for(leftright_index in 1:2){
      
      # leftright_index=1
      
      left_or_right = ifelse(leftright_index==1, "left", "right")
      sa_index = (RSprime_df_sub[,1]==sindex & RSprime_df_sub[,2]==left_or_right)
      
      n_sa = sum(sa_index)
      rvec = RSprime_df_sub[sa_index,3]
      sprimevec = RSprime_df_sub[sa_index,4]
      
      mu_saprime = muZ_prev_mat["right", sprimevec] # target policy (aprime) = right
      z_saprime = rnorm(n_sa, mean=mu_saprime, sd=sqrt(var_Z_prev))
      newz = rvec + gamma * z_saprime
      Z_samples[sa_index] = newz
      
      muZsa = muZ_prev_mat[left_or_right, sindex]
      mu_sub[sa_index] = muZsa
      nsa_sub = c(nsa_sub, n_sa)
    }
  }
  
  res <- optim(par=theta_prev, fn=Likelihood_singlestep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=100, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
  theta_prev =res$par
  parameter_matrix[,iteration+1] = theta_prev
}






