### Initialize Q-table

Qmat_left <- Qmat_right <- matrix(0, length(tau), S_size)
rownames(Qmat_left) <- rownames(Qmat_right) <- paste("tau=",tau, sep="")
colnames(Qmat_left) <- colnames(Qmat_right) <- paste("S=", 1:S_size, sep="")
Qmats = list(left=Qmat_left, right=Qmat_right)


### Integrate the Data List into a single dataframe.

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

RSprime_df_mix = RSprime_df[sample(x=1:N, size=N, replace = F),] # additional step (not in FLE)


for(time in 1:N){
  
  # time=1
  
  obs = RSprime_df_mix[time,]
  
  s_prev = obs[1][1,1]
  action = obs[2][1,1]
  reward = obs[3][1,1]
  s_post = obs[4][1,1]
  
  for(i in 1:update_per_sample){
    delta <- reward + gamma * sample(x=Qmats$right[,s_post], size=1) - Qmats[[action]][,s_prev] # a'=right. a=action
    inc <- (delta >= 0)
    dec <- (delta < 0)
    increment <- tau * alpha
    decrement <- (1-tau) * alpha
    Qmats[[action]][,s_prev] <- Qmats[[action]][,s_prev] + inc * increment - dec * decrement
  }
  
  # if(time %% 1000 == 0){ ## for plotting
  #   print(paste0("time = ",time, " / ", N))
  #   par(mfrow=c(1,2), oma=c(0,0,2,0))
  #   matplot(1:S_size, t(Qmats$left), main = "Q(s,left)")
  #   matplot(1:S_size, t(Qmats$right), main = "Q(s,right)")
  #   mtext(paste0("time = ", cumulative_time + time), outer=T, cex=2, line=-1)
  # }
}


