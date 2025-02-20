Energy_singlestep = function(theta_vec){
  
  # theta_vec = theta_initial
  # theta_vec=theta_pseudo
  
  beta0_L = theta_vec[1]
  beta0_R = theta_vec[2]
  beta1 = theta_vec[3]
  sig2 = theta_vec[4]
  
  muZ_left_candidate_vec = beta0_L + beta1 * svec
  muZ_right_candidate_vec = beta0_R + beta1 * svec
  muZ_candidate_mat = rbind(muZ_left_candidate_vec, muZ_right_candidate_vec)
  var_Z_candidate = sig2 / (1-gamma^2)
  
  energy_matrix = matrix(NA, 2, S_size)
  rownames(energy_matrix) = c("left", "right")
  colnames(energy_matrix) = paste0("S",1:S_size)
  
  for(sindex in 1:S_size){
    
    # sindex=1
    
    for(leftright in 1:2){
      
      # leftright=1
      
      LHS_up = muZ_candidate_mat[leftright,sindex] + 3*sqrt(var_Z_candidate) 
      LHS_down = muZ_candidate_mat[leftright,sindex] - 3*sqrt(var_Z_candidate) 
      
      rsprime = RSprime[[sindex]][[leftright]]
      sprime_index = unique(rsprime[,2]) # sprime
      r = rsprime[,1]
      
      RHS_up = gamma * (muZ_right_candidate_vec[sprime_index] + 3*sqrt(var_Z_candidate)) + max(r) # R(s,L) + gamma * Z(Sprime, R)
      RHS_down = gamma * (muZ_right_candidate_vec[sprime_index] - 3*sqrt(var_Z_candidate)) + min(r) 
      
      topgrid = max(LHS_up, RHS_up)
      bottomgrid = min(LHS_down, RHS_down)
      grids = seq(from=bottomgrid, to=topgrid, length.out=gridno) # (s,a)-dependent
      
      ## Bellman residual
      
      fX = dnorm(x=grids, mean=muZ_candidate_mat[leftright,sindex], sd=sqrt(var_Z_candidate))
      newx = outer(grids, r, "-") / gamma
      fY_individual = dnorm(newx, mean=muZ_right_candidate_vec[sprime_index], sd=sqrt(var_Z_candidate)) / gamma
      fY=rowMeans(fY_individual)
      
      # plot(grids, fX)
      # plot(grids, fY)
      # sum(fX) * diff(grids)[1]
      # sum(fY) * diff(grids)[1]
      
      fX_subt_fY = fX-fY
      second = outer(fX_subt_fY, fX_subt_fY, "*")
      first = abs(outer(grids, grids, "-"))
      energy = sum(-first * second) * diff(grids)[1] * diff(grids)[1]
      energy_matrix[leftright, sindex] = energy
    }
  }
  
  return(sum(energy_matrix * Nsa_leftright/N))
}


Energy_multistep <- function(theta_vec){
  
  # theta_vec=theta_initial
  # theta_vec=theta_pseudo
  # theta_vec = res$par
  
  beta0_L = theta_vec[1]
  beta0_R = theta_vec[2]
  beta1 = theta_vec[3]
  sig2 = theta_vec[4]
  
  muZ_left_candidate_vec = beta0_L + beta1 * svec
  muZ_right_candidate_vec = beta0_R + beta1 * svec
  muZ_candidate_mat = rbind(muZ_left_candidate_vec, muZ_right_candidate_vec)
  var_Z_candidate = sig2 / (1-gamma^2)
  
  energy_state_matrix = matrix(NA, 2, S_size)
  colnames(energy_state_matrix) = as.character(1:S_size)
  rownames(energy_state_matrix) = c("left", "right")
  
  energy_matrix = matrix(NA, 2, S_size)
  
  for (state_index in 1:S_size){
    
    # state_index=1

    for(leftright_index in 1:2){
      
      # leftright_index=1
      
      ### Set grids common for LHS and RHS
      
      mu_LHS = muZ_candidate_mat[leftright_index, state_index]
      
      zmax_LHS = mu_LHS + 3* sqrt(var_Z_candidate)
      zmin_LHS = mu_LHS - 3* sqrt(var_Z_candidate)
      
      YSm = YSm.list[[state_index]][[leftright_index]]
      Sm_candidate_vec = unique(YSm[,2])
      # Sm_candidate_vec = c(8,9,10) # 시험용
      zrange_RHS = c()
      for(Sm_candidate_index in 1:length(Sm_candidate_vec)){
        # Sm_candidate_index=1
        mu_RHS = muZ_right_candidate_vec[Sm_candidate_vec[Sm_candidate_index]] # target policy: right
        shrinked_mu_RHS = gamma^mstep * mu_RHS    
        shrinked_var_RHS = gamma^(mstep*2) * var_Z_candidate
        shrinked_zmax_RHS = shrinked_mu_RHS + 3*sqrt(shrinked_var_RHS)
        shrinked_zmin_RHS = shrinked_mu_RHS - 3*sqrt(shrinked_var_RHS)
        zrange_RHS = c(zrange_RHS, range(YSm[,1]) + c(shrinked_zmin_RHS, shrinked_zmax_RHS))
      }
      zmax_RHS = max(zrange_RHS)    
      zmin_RHS = min(zrange_RHS)    
      zmax = max(zmax_LHS, zmax_RHS)
      zmin = min(zmin_LHS, zmin_RHS)
      
      zgrids = seq(from=zmin, to=zmax, length.out=gridno)
      
      
      ### LHS and RHS density
      
      pdf_LHS = dnorm(x=zgrids, mean = mu_LHS, sd=sqrt(var_Z_candidate))
      
      newz = outer(zgrids, YSm[,1], "-") / gamma^mstep
      pdf_RHS_individual = t(dnorm(t(newz), mean=muZ_right_candidate_vec[YSm[,2]], sd=sqrt(var_Z_candidate))) / gamma^mstep
      pdf_RHS = rowMeans(pdf_RHS_individual)

      # sum(pdf_LHS) * diff(zgrids)[1]
      # sum(pdf_RHS) * diff(zgrids)[1]
      # plot(zgrids,pdf_LHS)
      # plot(zgrids,pdf_RHS)

      density_diff = pdf_LHS - pdf_RHS
      subtraction_norm=sqrt(abs(outer(zgrids, zgrids, "-")))
      energy_sa = -sum(subtraction_norm * outer(density_diff, density_diff)) * diff(zgrids)[1] * diff(zgrids)[1]
      
      energy_matrix[leftright_index, state_index] = energy_sa
    }
  }
  
  energybar = energy_matrix * Nsa_leftright / N
  return(sum(energybar))
}


Likelihood_singlestep <- function(theta_candidate){
  
  # theta_candidate = theta_prev
  
  beta0_L_candidate = theta_candidate[1]
  beta0_R_candidate = theta_candidate[2]
  beta1_candidate = theta_candidate[3]
  sig2_candidate = theta_candidate[4]
  
  muZ_left_candidate_vec = beta0_L_candidate + beta1_candidate * svec
  muZ_right_candidate_vec = beta0_R_candidate + beta1_candidate * svec
  muZ_candidate_mat = rbind(muZ_left_candidate_vec, muZ_right_candidate_vec)
  rownames(muZ_candidate_mat) = c("left", "right")
  var_Z_candidate = sig2_candidate / (1-gamma^2)
  
  mu_candidates = rep(c(muZ_candidate_mat), c(nsa_sub)) 
  log_likelihood = sum(dnorm(Z_samples, mean=mu_candidates, sd=sqrt(var_Z_candidate), log=TRUE))
  
  return(-log_likelihood)
}



Inaccuracy <- function(theta_vec){
  
  muZ_left = A_true*p_true^(svec-1) * (1-(gamma*p_true)^(S_size-svec+2)) / (1-gamma*p_true) + gamma^(S_size-svec+2) / (1-gamma) * mu_last # E(Z(s,L))
  muZ_right = A_true*p_true^(svec+1) * (1-(gamma*p_true)^(S_size-svec)) / (1-gamma*p_true) + gamma^(S_size-svec) / (1-gamma) * mu_last # E(Z(s,R))
  muZ_left[1] = gamma * muZ_right[1] + muR_left[1] # exception for (s,a)=(1,L)
  
  left_upper = muZ_left + 3*sqrt(var_Z_true)
  left_below = muZ_left - 3*sqrt(var_Z_true)
  right_upper = muZ_right + 3*sqrt(var_Z_true)
  right_below = muZ_right - 3*sqrt(var_Z_true)
  
  
  beta0_L = theta_vec[1]
  beta0_R = theta_vec[2]
  beta1 = theta_vec[3]
  sig2 = theta_vec[4]
  
  muZ_left_candidate_vec = beta0_L + beta1 * svec
  muZ_right_candidate_vec = beta0_R + beta1 * svec
  var_Z_candidate = sig2 / (1-gamma^2)
  
  energy_state_matrix = matrix(NA, 2, S_size)
  
  for (state_index in 1:S_size){
    
    # state_index = 1
    
    ### Left
    
    muZ_left_candidate = muZ_left_candidate_vec[state_index]
    Z_left_candidate_upper = muZ_left_candidate + 3 * sqrt(var_Z_candidate)
    Z_left_candidate_below = muZ_left_candidate - 3 * sqrt(var_Z_candidate)
    Z_left_true_upper = left_upper[state_index]
    Z_left_true_below = left_below[state_index]
    
    zmin_left = min(Z_left_candidate_below, Z_left_true_below)
    zmax_left = max(Z_left_candidate_upper, Z_left_true_upper)
    z_grids_left = seq(from=zmin_left, to=zmax_left, length.out = gridno)
    
    pdf_left_candidate = dnorm(x=z_grids_left, mean=muZ_left_candidate, sd=sqrt(var_Z_candidate))
    pdf_left_true = dnorm(x=z_grids_left, mean=muZ_left[state_index], sd=sqrt(var_Z_true))
    
    # plot(z_grids_left, pdf_left_candidate)
    # points(z_grids_left, pdf_left_true)
    # sum(pdf_left_candidate) * diff(z_grids_left)[1]
    # sum(pdf_left_true) * diff(z_grids_left)[1]
    
    norm_value_left = abs(outer(z_grids_left, z_grids_left, "-"))
    pdf_diff_left = pdf_left_candidate - pdf_left_true
    pdf_product_left = outer(pdf_diff_left, pdf_diff_left, "*")
    
    energy_state_left = - sum(norm_value_left * pdf_product_left) * diff(z_grids_left)[1] * diff(z_grids_left)[1]
    energy_state_matrix[1,state_index] = energy_state_left
    
    
    ### Right
    
    muZ_right_candidate = muZ_right_candidate_vec[state_index]
    Z_right_candidate_upper = muZ_right_candidate + 3 * sqrt(var_Z_candidate)
    Z_right_candidate_below = muZ_right_candidate - 3 * sqrt(var_Z_candidate)
    Z_right_true_upper = right_upper[state_index]
    Z_right_true_below = right_below[state_index]
    
    zmin_right = min(Z_right_candidate_below, Z_right_true_below)
    zmax_right = max(Z_right_candidate_upper, Z_right_true_upper)
    z_grids_right = seq(from=zmin_right, to=zmax_right, length.out = gridno)
    
    pdf_right_candidate = dnorm(x=z_grids_right, mean=muZ_right_candidate, sd=sqrt(var_Z_candidate))
    pdf_right_true = dnorm(x=z_grids_right, mean=muZ_right[state_index], sd=sqrt(var_Z_true))
    
    # plot(z_grids_right, pdf_right_candidate)
    # points(z_grids_right, pdf_right_true)
    # sum(pdf_right_candidate) * diff(z_grids_right)[1]
    # sum(pdf_right_true) * diff(z_grids_right)[1]
    
    norm_value_right = abs(outer(z_grids_right, z_grids_right, "-"))
    pdf_diff_right = pdf_right_candidate - pdf_right_true
    pdf_product_right = outer(pdf_diff_right, pdf_diff_right, "*")
    
    energy_state_right = - sum(norm_value_right * pdf_product_right) * diff(z_grids_right)[1] * diff(z_grids_right)[1]
    energy_state_matrix[2,state_index] = energy_state_right
    
  }
  
  energybar = mean(energy_state_matrix) # fair probability
  return(energybar)
}


Disparity_thetas <- function(theta_vec1, theta_vec2){ # special case: Inaccuracy function
  
  beta0_L_cand1 = theta_vec1[1]
  beta0_R_cand1 = theta_vec1[2]
  beta1_cand1 = theta_vec1[3]
  sig2_cand1 = theta_vec1[4]
  
  muZ_left_cand1_vec = beta0_L_cand1 + beta1_cand1 * svec
  muZ_right_cand1_vec = beta0_R_cand1 + beta1_cand1 * svec
  var_Z_cand1 = sig2_cand1 / (1-gamma^2)
  
  left_upper = muZ_left_cand1_vec + 3*sqrt(var_Z_cand1)
  left_below = muZ_left_cand1_vec - 3*sqrt(var_Z_cand1)
  right_upper = muZ_right_cand1_vec + 3*sqrt(var_Z_cand1)
  right_below = muZ_right_cand1_vec - 3*sqrt(var_Z_cand1)
  
  
  beta0_L_cand2 = theta_vec2[1]
  beta0_R_cand2 = theta_vec2[2]
  beta1_cand2 = theta_vec2[3]
  sig2_cand2 = theta_vec2[4]
  
  muZ_left_cand2_vec = beta0_L_cand2 + beta1_cand2 * svec
  muZ_right_cand2_vec = beta0_R_cand2 + beta1_cand2 * svec
  var_Z_cand2 = sig2_cand2 / (1-gamma^2)
  
  energy_matrix = matrix(NA, 2, S_size)
  rownames(energy_matrix) = c("left", "right")
  colnames(energy_matrix) = paste0("S",1:S_size)
  
  
  for (state_index in 1:S_size){
    
    # state_index = 1
    
    ### Left
    
    muZ_left_cand2 = muZ_left_cand2_vec[state_index]
    Z_left_cand2_upper = muZ_left_cand2 + 3 * sqrt(var_Z_cand2)
    Z_left_cand2_below = muZ_left_cand2 - 3 * sqrt(var_Z_cand2)
    Z_left_cand1_upper = left_upper[state_index]
    Z_left_cand1_below = left_below[state_index]
    
    zmin_left = min(Z_left_cand2_below, Z_left_cand1_below)
    zmax_left = max(Z_left_cand2_upper, Z_left_cand1_upper)
    z_grids_left = seq(from=zmin_left, to=zmax_left, length.out = gridno)
    
    pdf_left_cand2 = dnorm(x=z_grids_left, mean=muZ_left_cand2, sd=sqrt(var_Z_cand2))
    pdf_left_cand1 = dnorm(x=z_grids_left, mean=muZ_left_cand1_vec[state_index], sd=sqrt(var_Z_cand1))
    
    # plot(z_grids_left, pdf_left_cand2)
    # lines(z_grids_left, pdf_left_cand1)
    # sum(pdf_left_cand2) * diff(z_grids_left)[1]
    # sum(pdf_left_cand1) * diff(z_grids_left)[1]
    
    norm_value_left = abs(outer(z_grids_left, z_grids_left, "-"))
    pdf_diff_left = pdf_left_cand2 - pdf_left_cand1
    pdf_product_left = outer(pdf_diff_left, pdf_diff_left, "*")
    
    energy_state_left = - sum(norm_value_left * pdf_product_left) * diff(z_grids_left)[1] * diff(z_grids_left)[1]
    energy_matrix[1,state_index] = energy_state_left
    
    
    ### Right
    
    muZ_right_cand2 = muZ_right_cand2_vec[state_index]
    Z_right_cand2_upper = muZ_right_cand2 + 3 * sqrt(var_Z_cand2)
    Z_right_cand2_below = muZ_right_cand2 - 3 * sqrt(var_Z_cand2)
    Z_right_cand1_upper = right_upper[state_index]
    Z_right_cand1_below = right_below[state_index]
    
    zmin_right = min(Z_right_cand2_below, Z_right_cand1_below)
    zmax_right = max(Z_right_cand2_upper, Z_right_cand1_upper)
    z_grids_right = seq(from=zmin_right, to=zmax_right, length.out = gridno)
    
    pdf_right_cand2 = dnorm(x=z_grids_right, mean=muZ_right_cand2, sd=sqrt(var_Z_cand2))
    pdf_right_cand1 = dnorm(x=z_grids_right, mean=muZ_right_cand1_vec[state_index], sd=sqrt(var_Z_cand1))
    
    # plot(z_grids_right, pdf_right_cand2)
    # lines(z_grids_right, pdf_right_cand1)
    # sum(pdf_right_cand2) * diff(z_grids_right)[1]
    # sum(pdf_right_cand1) * diff(z_grids_right)[1]
    
    norm_value_right = abs(outer(z_grids_right, z_grids_right, "-"))
    pdf_diff_right = pdf_right_cand2 - pdf_right_cand1
    pdf_product_right = outer(pdf_diff_right, pdf_diff_right, "*")
    
    energy_state_right = - sum(norm_value_right * pdf_product_right) * diff(z_grids_right)[1] * diff(z_grids_right)[1]
    energy_matrix[2,state_index] = energy_state_right
    
  }
  
  energybarhat = sum(energy_matrix * Nsa_leftright/N)
  return(energybarhat)
}





Inaccuracy_parametric_Wasserstein <- function(theta_vec){
  
  # theta_vec = c(A_true, p_true)
  
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
  
  wasser_matrix = matrix(NA, 2, S_size)
  rownames(wasser_matrix) = c("left", "right")
  colnames(wasser_matrix) = paste0("S",1:S_size)
  
  for(state_index in 1:S_size){
    
    samples_left_true = rnorm(Wasserstein_samples, mean = muZ_left[state_index], sd=sqrt(var_Z_true))
    samples_left_candidate = rnorm(Wasserstein_samples, mean = muZ_left_candidate_vec[state_index], sd=sqrt(var_Z_candidate))
    
    samples_right_true = rnorm(Wasserstein_samples, mean = muZ_right[state_index], sd=sqrt(var_Z_true))
    samples_right_candidate = rnorm(Wasserstein_samples, mean = muZ_right_candidate_vec[state_index], sd=sqrt(var_Z_candidate))
    
    wasser_matrix["left", state_index] = transport::wasserstein1d(samples_left_true, samples_left_candidate)
    wasser_matrix["right", state_index] = transport::wasserstein1d(samples_right_true, samples_right_candidate)
  }
  
  wasser_mean = mean(wasser_matrix)
  return(wasser_mean)  
}




Inaccuracy_QRTD <- function(Qmats){
  
  # Qmats = Qmats_list[[sim_ind]] 
  
  muZ_left = A_true*p_true^(svec-1) * (1-(gamma*p_true)^(S_size-svec+2)) / (1-gamma*p_true) + gamma^(S_size-svec+2) / (1-gamma) * mu_last # E(Z(s,L))
  muZ_right = A_true*p_true^(svec+1) * (1-(gamma*p_true)^(S_size-svec)) / (1-gamma*p_true) + gamma^(S_size-svec) / (1-gamma) * mu_last # E(Z(s,R))
  muZ_left[1] = gamma * muZ_right[1] + muR_left[1] # exception for (s,a)=(1,L)
  
  left_upper = muZ_left + 3*sqrt(var_Z_true)
  left_below = muZ_left - 3*sqrt(var_Z_true)
  right_upper = muZ_right + 3*sqrt(var_Z_true)
  right_below = muZ_right - 3*sqrt(var_Z_true)
  
  Qmat_left = Qmats[["left"]]
  Qmat_right = Qmats[["right"]]
  
  energy_matrix = matrix(NA, 2, S_size)
  rownames(energy_matrix) = c("left", "right")
  colnames(energy_matrix) = paste0("S",1:S_size)
  
  
  # Qmat_left
  
  for (state_index in 1:S_size){
    
    # state_index = 1
    
    ### Left
    
    y_particles = Qmat_left[,state_index]; names(y_particles) = NULL
    Z_left_true_upper = left_upper[state_index]
    Z_left_true_below = left_below[state_index]
    z_grids_left = seq(from=Z_left_true_below, to=Z_left_true_upper, length.out = gridno)
    pdf_left_true = dnorm(x=z_grids_left, mean=muZ_left[state_index], sd=sqrt(var_Z_true))
    # sum(pdf_left_true) * diff(z_grids_left)[1]
    
    z_subtr_y = outer(z_grids_left, y_particles, "-")
    term1 = mean(crossprod(abs(z_subtr_y), pdf_left_true) * diff(z_grids_left)[1])
    
    z_subtr_z = outer(z_grids_left, z_grids_left, "-")
    term2 = sum(abs(z_subtr_z) * outer(pdf_left_true, pdf_left_true, "*")) * diff(z_grids_left)[1] * diff(z_grids_left)[1]
    
    y_subtr_y = outer(y_particles, y_particles, "-")
    term3 = mean(abs(y_subtr_y))
    
    energy_state_left = term1 * 2 - term2 - term3
    energy_matrix[1,state_index] = energy_state_left
    
    
    ### Right
    
    y_particles = Qmat_right[,state_index]; names(y_particles) = NULL
    Z_right_true_upper = right_upper[state_index]
    Z_right_true_below = right_below[state_index]
    z_grids_right = seq(from=Z_right_true_below, to=Z_right_true_upper, length.out = gridno)
    pdf_right_true = dnorm(x=z_grids_right, mean=muZ_right[state_index], sd=sqrt(var_Z_true))
    # sum(pdf_right_true) * diff(z_grids_right)[1]
    
    z_subtr_y = outer(z_grids_right, y_particles, "-")
    term1 = mean(crossprod(abs(z_subtr_y), pdf_right_true) * diff(z_grids_right)[1])
    
    z_subtr_z = outer(z_grids_right, z_grids_right, "-")
    term2 = sum(abs(z_subtr_z) * outer(pdf_right_true, pdf_right_true, "*")) * diff(z_grids_right)[1] * diff(z_grids_right)[1]
    
    y_subtr_y = outer(y_particles, y_particles, "-")
    term3 = mean(abs(y_subtr_y))
    
    energy_state_right = term1 * 2 - term2 - term3
    energy_matrix[2,state_index] = energy_state_right
  }
  
  energybar = mean(energy_matrix)
  return(energybar)
}



Inaccuracy_QRTD_Wasserstein <- function(Qmats){
  
  # Qmats = Qmats_list[[sim_ind]]
  
  muZ_left = A_true*p_true^(svec-1) * (1-(gamma*p_true)^(S_size-svec+2)) / (1-gamma*p_true) + gamma^(S_size-svec+2) / (1-gamma) * mu_last # E(Z(s,L))
  muZ_right = A_true*p_true^(svec+1) * (1-(gamma*p_true)^(S_size-svec)) / (1-gamma*p_true) + gamma^(S_size-svec) / (1-gamma) * mu_last # E(Z(s,R))
  muZ_left[1] = gamma * muZ_right[1] + muR_left[1] # exception for (s,a)=(1,L)
  
  wasser_matrix = matrix(NA, 2, S_size)
  rownames(wasser_matrix) = c("left", "right")
  colnames(wasser_matrix) = paste0("S",1:S_size)
  
  for(state_index in 1:S_size){
    
    # state_index=1
    
    samples_left_true = rnorm(Wasserstein_samples, mean = muZ_left[state_index], sd=sqrt(var_Z_true))
    samples_left_candidate = sample(x=Qmats[["left"]][,state_index], size=Wasserstein_samples, replace=T)    
    
    samples_right_true = rnorm(Wasserstein_samples, mean = muZ_right[state_index], sd=sqrt(var_Z_true))
    samples_right_candidate = sample(x=Qmats[["right"]][,state_index], size=Wasserstein_samples, replace=T)    
    
    wasser_matrix["left", state_index] = transport::wasserstein1d(samples_left_true, samples_left_candidate)
    wasser_matrix["right", state_index] = transport::wasserstein1d(samples_right_true, samples_right_candidate)
  }
  
  wasser_mean = mean(wasser_matrix)
  return(wasser_mean)  
}







