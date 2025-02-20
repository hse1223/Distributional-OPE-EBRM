## Integrate the Data List into a single dataframe.

RSprime_df = vector("list", S_size)
for(sindex in 1:S_size){
  # sindex=5
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

# Nprime=sum(sapply(RSprime, function(x) sapply(x, nrow))) # 지우기. (임시) just for the case of Nsa_right[5]=0
# index_mixed = sample(x=1:Nprime, size=Nprime, replace=FALSE) # 지우기. (임시)

A_vec = p_vec = rep(NA, partition+1)
A_vec[1] = theta_prev[1]; p_vec[1]=theta_prev[2]


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
  
  
  A_prev = theta_prev[1]
  p_prev = theta_prev[2]  
  
  muZ_left_prev_vec = A_prev*p_prev^(svec-1) * (1-(gamma*p_prev)^(S_size-svec+2)) / (1-gamma*p_prev) + gamma^(S_size-svec+2) / (1-gamma) * mu_last
  muZ_right_prev_vec = A_prev*p_prev^(svec+1) * (1-(gamma*p_prev)^(S_size-svec)) / (1-gamma*p_prev) + gamma^(S_size-svec) / (1-gamma) * mu_last
  # muZ_left_prev_vec[1] = gamma * muZ_right_prev_vec[1] + muR_left[1] # exception for (s,a)=(1,L)
  muZ_left_prev_vec[1] = gamma * muZ_right_prev_vec[1] + A_prev # exception for (s,a)=(1,L)
  
  muZ_prev_mat = rbind(muZ_left_prev_vec, muZ_right_prev_vec)
  rownames(muZ_prev_mat) = c("left", "right")
  
  Z_samples = rep(NA, nrow(RSprime_df_sub))
  # mu_sub = rep(NA, nrow(RSprime_df_sub))
  nsa_sub = c()
  
  for(sindex in 1:S_size){
    
    # sindex=5
    
    for(leftright_index in 1:2){
      
      # leftright_index=2
      
      left_or_right = ifelse(leftright_index==1, "left", "right")
      sa_index = (RSprime_df_sub[,1]==sindex & RSprime_df_sub[,2]==left_or_right)
      
      n_sa = sum(sa_index)
      rvec = RSprime_df_sub[sa_index,3]
      sprimevec = RSprime_df_sub[sa_index,4]
      
      mu_saprime = muZ_prev_mat["right", sprimevec] # target policy (aprime) = right
      z_saprime = rnorm(n_sa, mean=mu_saprime, sd=sqrt(var_Z_true))
      newz = rvec + gamma * z_saprime
      Z_samples[sa_index] = newz
      
      # muZsa = muZ_prev_mat[left_or_right, sindex]
      # mu_sub[sa_index] = muZsa
      nsa_sub = c(nsa_sub, n_sa)
    }
  }
  
  res <- optim(par=theta_prev, fn=Likelihood_singlestep, method="L-BFGS-B", control=list(trace=TRUE, REPORT=100, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
  theta_prev =res$par
  
  A_vec[iteration+1] = res$par[1]
  p_vec[iteration+1] = res$par[2]
  
}


