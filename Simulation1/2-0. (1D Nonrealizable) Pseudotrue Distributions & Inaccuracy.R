rm(list=ls())
# dev.off()

##### Parameter Setting

### Environment
gamma = 0.50
# gamma = 0.99
mu_last = 0
gridno = 100


### Setting
S_size=30; A_true = 100; sig2_true = 20; p_true=0.9 # setting2: reasonable setting


##### True Distribution

svec=c(1:S_size)
muR_left = A_true * p_true^(svec - 1) # E(R(s,L))
muR_right = A_true * p_true^(svec + 1); muR_right[S_size] = mu_last # E(R(s,R))

muZ_left = A_true*p_true^(svec-1) * (1-(gamma*p_true)^(S_size-svec+2)) / (1-gamma*p_true) + gamma^(S_size-svec+2) / (1-gamma) * mu_last # E(Z(s,L))
muZ_right = A_true*p_true^(svec+1) * (1-(gamma*p_true)^(S_size-svec)) / (1-gamma*p_true) + gamma^(S_size-svec) / (1-gamma) * mu_last # E(Z(s,R))
muZ_left[1] = gamma * muZ_right[1] + muR_left[1] # exception for (s,a)=(1,L)
var_Z = sig2_true/(1-gamma^2)

left_upper = muZ_left + 3*sqrt(var_Z)
left_below = muZ_left - 3*sqrt(var_Z)
right_upper = muZ_right + 3*sqrt(var_Z)
right_below = muZ_right - 3*sqrt(var_Z)
ymax = max(c(left_upper, right_upper))
ymin = min(c(left_below, right_below))



##### Inaccuracy Function

# thetavec = c("beta0_L" = 600, "beta0_R" = 700, "beta1" = -50, sig2 = 20) # 나중에 initial_value도 대체하자. population quantity에 의존하게끔.

Inaccuracy <- function(thetavec){
  
  beta0_L = thetavec[1]
  beta0_R = thetavec[2]
  beta1 = thetavec[3]
  sig2 = thetavec[4]
  
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
    pdf_left_true = dnorm(x=z_grids_left, mean=muZ_left[state_index], sd=sqrt(var_Z))
    
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
    pdf_right_true = dnorm(x=z_grids_right, mean=muZ_right[state_index], sd=sqrt(var_Z))
    
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


##### Pseudo-True Parameter Values

theta_initial = c("beta0_L" = 600, "beta0_R" = 700, "beta1" = -50, sig2 = 20) # 나중에 initial_value도 대체하자. population quantity에 의존하게끔.
# Inaccuracy(theta_initial)

lower_bounds = c(0,0,-200, 1)
upper_bounds = c(2000, 2000, 100, 1000)

res <- optim(par=theta_initial, fn=Inaccuracy, method="L-BFGS-B", control=list(trace=TRUE, REPORT=1, factr=100, maxit=200), lower=lower_bounds, upper=upper_bounds)
(theta_pseudo = res$par)
(inaccuracy = res$value)



##### Plot Pseudo-True Distributions

beta0_L_pseudo = theta_pseudo[1]
beta0_R_pseudo = theta_pseudo[2]
beta1_pseudo = theta_pseudo[3]
sig2_pseudo = theta_pseudo[4]

muZ_left_pseudo_vec = beta0_L_pseudo + beta1_pseudo * svec
muZ_right_pseudo_vec = beta0_R_pseudo + beta1_pseudo * svec
var_Z_pseudo = sig2_pseudo / (1-gamma^2)

left_upper_pseudo = muZ_left_pseudo_vec + 3*sqrt(var_Z_pseudo)
left_below_pseudo = muZ_left_pseudo_vec - 3*sqrt(var_Z_pseudo)
right_upper_pseudo = muZ_right_pseudo_vec + 3*sqrt(var_Z_pseudo)
right_below_pseudo = muZ_right_pseudo_vec - 3*sqrt(var_Z_pseudo)


# plot(svec, muZ_left_pseudo_vec, ylim=c(ymin, ymax), main="Pseudo-true Zpi (Return) Distributions", ylab="")
plot(svec, muZ_left_pseudo_vec, ylim=c(ymin, ymax), main=expression(paste("Best approximation Z(s,a;",tilde(theta),")")), xlab="state", ylab="value", type="n")
for (sindex in svec) {
  # sindex=1
  lines(c(sindex, sindex), c(left_upper_pseudo[sindex], left_below_pseudo[sindex]), col="red", lwd=3)
}
lines(cbind(svec, muZ_left_pseudo_vec), col="red", lwd=3, lty="dotted")
points(svec, muZ_left_pseudo_vec, ylim=c(ymin, ymax), pch=16, cex=1.2, col="red")
for (sindex in svec) {
  # sindex=1
  lines(c(sindex+0.05, sindex+0.05), c(right_upper_pseudo[sindex], right_below_pseudo[sindex]), col="blue", lwd=3)
}
lines(cbind(svec, muZ_right_pseudo_vec), col="blue", lwd=3, lty="dotted")
points(svec, muZ_right_pseudo_vec, ylim=c(ymin, ymax), pch=16, cex=1.2, col="blue")

legends = c("a=(-1)", "a=   1")
legend("topright", legend = legends, col=c("red", "blue"), lty=1)

mtext(paste0("variance=", sig2_true, " & gamma=", gamma), line=0.5)



# ##### Further Confirm that it is indeed global minimizer
# 
# beta0_L_vec = theta_pseudo["beta0_L"] + seq(from=-200, to=200, length.out=51)
# beta0_R_vec = theta_pseudo["beta0_R"] + seq(from=-200, to=200, length.out=51)
# beta1_vec = theta_pseudo["beta1"] + seq(from=-30, to=30, length.out=51)
# sig2_vec = theta_pseudo["sig2"] + seq(from=-100, to=100, length.out=51)
# 
# Big_Parameter_Matrix = expand.grid(beta0_L_vec, beta0_R_vec, beta1_vec, sig2_vec)
# # which(Big_Parameter_Matrix[,1] == theta_pseudo["beta0_L"] & Big_Parameter_Matrix[,2] == theta_pseudo["beta0_R"] & Big_Parameter_Matrix[,3] == theta_pseudo["beta1"] & Big_Parameter_Matrix[,4] == theta_pseudo["sig2"] )
# 
# 
# library(doParallel)
# nworkers <- parallel::detectCores() / 2 # Initialize the cluster.
# cl <- parallelly::makeClusterPSOCK(nworkers)
# doParallel::registerDoParallel(cl)
# 
# tictoc::tic()
# Inaccuracy_vec <- foreach (row_index = 1:nrow(Big_Parameter_Matrix)) %dopar% {
#   theta_candidate = Big_Parameter_Matrix[row_index,]
#   theta_candidate = c(theta_candidate[1,1], theta_candidate[1,2], theta_candidate[1,3], theta_candidate[1,4])
#   Inaccuracy(theta_candidate)
# }
# tictoc::toc()
# stopCluster(cl)
# Inaccuracy_vec = unlist(Inaccuracy_vec)
# 
# Big_Parameter_Matrix[which.min(Inaccuracy_vec),]
# theta_pseudo

