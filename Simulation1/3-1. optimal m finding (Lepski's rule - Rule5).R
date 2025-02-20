### Realizable 

rm(list=ls())

sig2=20; N_vec = c(500, 1000, 2000, 5000, 10000, 20000)
# sig2=5000; N_vec = c(2000, 5000, 10000, 20000, 50000, 1e+05)

# constant_multi=1.96 # for bandwidth 
constant_multi=1 # for bandwidth

filenames = paste0("results/optimal_m/realizable_var", sig2, "/EBRM_realizable_disparity_var", sig2, "N", N_vec, "Trial1to50.txt")
par(mfrow=c(2,3), mar= c(5.1, 4.1, 2.1, 2.1), oma=c(0,0,2,0))



for(i in 1:length(N_vec)){
  
  # i=1
  
  disparity_table=read.table(filenames[i])
  
  # disparity_table = disparity_table[,1:7]
  
  steplevels = as.numeric(gsub("m.","", colnames(disparity_table)))
  
  mean_values = apply(disparity_table, 2, mean)
  sd_values = apply(disparity_table, 2, sd)
  
  upper_values = mean_values + constant_multi*sd_values
  lower_values = mean_values - constant_multi*sd_values
  ylim=range(c(upper_values, lower_values, disparity_table))
  
  matplot(steplevels, t(disparity_table), ylim=ylim, xlab="step level (m)", ylab="Disparity")
  # mtext(paste0("N=", N_vec[i])  , outer=F, line=0.3)
  
  lines(steplevels,upper_values, col=2, lty="dotted")
  lines(steplevels,lower_values, col=2, lty="dotted")
  lines(steplevels,mean_values, lwd=3, col=2)
  
  mean_diff=c(mean_values[1], diff(mean_values))
  
  # stopindex=which(cumsum(mean_diff<0)==3) # HARD-CODED: 3 IS ARBITRARY.
  stopindex=which(cumsum(mean_diff<0)==1) # HARD-CODED: 3 IS ARBITRARY.
  if(length(stopindex)==0){
    stopindex = length(steplevels)
  }
  steplevels_consider = steplevels[1:stopindex]
  
  
  
  
  lower_upper = rbind(lower_values, upper_values)
  
  became_empty = FALSE
  for(ind in length(steplevels_consider):1){
    
    # ind = length(steplevels_consider)
    
    max_lower = max(lower_upper[1,ind:length(steplevels_consider)])
    min_upper = min(lower_upper[2,ind:length(steplevels_consider)])
    
    became_empty = min_upper < max_lower
    if(became_empty){
      break
    }
  }
  
  if (became_empty){
    chosen_m = steplevels_consider[ind]
  } else {
    chosen_m = 1
  }
  abline(v=chosen_m)
  
  mtext(paste0("N=", N_vec[i],": m=", chosen_m)  , outer=F, line=0.3)
}

mtext(paste0("Realizable: sig2=", sig2), outer=T, cex=1.5, line=-0.1)


### Non-realizable 

rm(list=ls())

# gamma=0.50; N_vec = c(2000, 3000, 5000, 10000)
gamma=0.99; N_vec = c(2000, 3000, 5000, 10000)

# constant_multi=1.96 # for bandwidth
constant_multi=1 # for bandwidth

filenames = paste0("results/optimal_m/nonrealizable_gamma", gamma*100, "/EBRM_nonrealizable_disparity_gamma", gamma*100, "N", N_vec, "Trial1to50.txt")
par(mfrow=c(1,4), mar= c(5.1, 4.1, 2.1, 2.1), oma=c(0,0,2,0))


for(i in 1:length(N_vec)){
  
  # i=1
  
  disparity_table=read.table(filenames[i])
  
  
  # if(gamma==0.99){
  #   disparity_table = disparity_table[,1:14]
  # } 
  
  if(gamma==0.99){
    disparity_table = disparity_table[,1:13]
  } else {
    disparity_table = disparity_table[,1:3]
  }
  
  steplevels = as.numeric(gsub("m.","", colnames(disparity_table)))
  
  mean_values = apply(disparity_table, 2, mean)
  sd_values = apply(disparity_table, 2, sd)
  
  upper_values = mean_values + constant_multi*sd_values
  lower_values = mean_values - constant_multi*sd_values

  ylim=range(c(upper_values, lower_values, disparity_table))
  if(gamma==0.99){
    # ylim=c(0,max(ylim))
    ylim=c(3000,8000)
  } 
  
  matplot(steplevels, t(disparity_table), ylim=ylim, xlab="step level (m)", ylab="Disparity")
  
  lines(steplevels,upper_values, col=2, lty="dotted")
  lines(steplevels,lower_values, col=2, lty="dotted")
  lines(steplevels,mean_values, lwd=3, col=2)
  
  
  lower_upper = rbind(lower_values, upper_values)
  
  became_empty = FALSE
  for(ind in length(steplevels):1){
    
    # ind = length(steplevels)
    
    max_lower = max(lower_upper[1,ind:length(steplevels)])
    min_upper = min(lower_upper[2,ind:length(steplevels)])
    
    became_empty = min_upper < max_lower
    if(became_empty){
      break
    }
  }
  
  if (became_empty){
    chosen_m = steplevels[ind]
  } else {
    chosen_m = 1
  }
  abline(v=chosen_m)
  
  mtext(paste0("N=", N_vec[i],": m=", chosen_m)  , outer=F, line=0.3)
  
  print(round(sd_values,3))
}

mtext(paste0("Non-realizable: gamma=", gamma), outer=T, cex=1.5, line=-0.2)

























