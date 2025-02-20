### Generate Data.

svec=c(1:S_size)
muR_left = A_true * p_true^(svec - 1) # E(R(s,L))
muR_right = A_true * p_true^(svec + 1); muR_right[S_size] = mu_last # E(R(s,R))

RSprime = vector("list", S_size)
names(RSprime) = paste0("S",1:S_size)

Nsa = table(c(sample(x=1:(2*S_size), size=N, replace = T)))
Nsa_left = Nsa[as.character(1:S_size)]
names(Nsa_left) = as.character(1:S_size)
Nsa_left[is.na(Nsa_left)] = 0
Nsa_right = Nsa[as.character(1:S_size+S_size)]

# Nsa_right[5]=0 # IF YOU WANT TO TEST A CASE WHERE WE HAVE N(s,a)=0

names(Nsa_left) = as.character(1:S_size)
Nsa_right[is.na(Nsa_right)] = 0
Nsa_leftright = rbind(Nsa_left, Nsa_right)
rownames(Nsa_leftright) <- c("left", "right")


for(sindex in 1:S_size){
  
  # sindex=1
  
  if(Nsa_left[sindex]==0){
    reward_left = 0
    RSprime_left = cbind(Reward=reward_left, Sprime=sindex)
  } else {
    reward_left = rnorm(Nsa_left[sindex], mean=muR_left[sindex], sd=sqrt(sig2_true)) # Caution: SD, not variance.
    RSprime_left = cbind(Reward=reward_left, Sprime=sindex-1)
  }
  
  if(Nsa_right[sindex]==0){
    reward_right = 0
    RSprime_right = cbind(Reward=reward_right, Sprime=sindex)
  } else {
    reward_right = rnorm(Nsa_right[sindex], mean=muR_right[sindex], sd=sqrt(sig2_true)) # Caution: SD, not variance.
    RSprime_right = cbind(Reward=reward_right, Sprime=sindex+1)
  }
  
  RSprime[[sindex]] = list(left=RSprime_left, right=RSprime_right)
}
RSprime$S1$left[,2] = 1 # no terminal state 
RSprime[[S_size]]$right[,2] = S_size # no terminal state 


# ### Form it as into a data-frame.
# 
# RSprime_new = vector("list", S_size)
# for(sindex in 1:S_size){
#   # sindex=1
#   SL_data = data.frame(S=sindex, A="left",RSprime[[sindex]]$left)
#   SR_data = data.frame(S=sindex, A="right",RSprime[[sindex]]$right)
#   RSprime_new[[sindex]] = rbind(SL_data, SR_data)
# }
# RSprime = do.call(rbind, RSprime_new)
# 
# if(sum(is.na(RSprime$Reward)) > 0){
#   RSprime = RSprime[-which(is.na(RSprime$Reward)),]
# }







