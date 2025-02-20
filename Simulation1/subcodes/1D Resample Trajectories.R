##### Resampling Trajectories

### Sample Initial (s,a)

Msa = sample(x=1:(S_size*2), size = M_size, replace=TRUE, prob=c(Nsa_left, Nsa_right) / N)
Msa_vec <- rep(NA, S_size*2)
names(Msa_vec) = 1:(S_size*2)
for(i in 1:(S_size*2)){
  Msa_vec[i] = sum(Msa==i)
}
Msa_vec[is.na(Msa_vec)] = 0
Msa_leftright = matrix(Msa_vec, 2, S_size, byrow = T)
rownames(Msa_leftright) <- c("left", "right")
colnames(Msa_leftright) <- c(1:S_size)
# Msa_leftright[2,5]=0 # 혹시 시험하고 싶으면.


### Generate (Y,Sm) for each (s,a)

YSm.list <- vector("list", S_size)
names(YSm.list) = paste0("S",1:S_size)
gammas = cumprod(rep(gamma, mstep)) / gamma  

for(sindex in 1:S_size){
  
  # sindex=5 # If blank in Nsa, then there should be 0-trajectories.
  
  Ysm.sublist <- vector("list", 2)
  names(Ysm.sublist) = c("left", "right")
  
  for(leftright_index in 1:2){
    
    # leftright_index=2
    
    Msa = Msa_leftright[leftright_index,sindex] # Left
    
    if (Msa!=0){
      R_mat = matrix(NA, Msa, mstep)  
      Sm_vec = rep(NA, Msa)
      
      for(resample_index in 1:Msa){
        
        # resample_index=1
        
        s_prev = sindex # initial s,a 
        a_prev = ifelse(leftright_index==1, "left", "right")  
        
        for(iter in 1:mstep){
          # iter=6
          rsprime = RSprime[[s_prev]][[a_prev]][sample(x=1:nrow(RSprime[[s_prev]][[a_prev]]), size=1),]
          R_mat[resample_index,iter] = rsprime[1]
          s_prev = rsprime[2]
          a_prev = "right"# target policy
        }
        Sm_vec[resample_index]=s_prev
      }
      Yvec = c(R_mat %*% gammas)
      YSm_mat = cbind(Ym=Yvec, Sm=Sm_vec)
    } else {
      YSm_mat = cbind(Ym=0, Sm=sindex)
    }
    Ysm.sublist[[leftright_index]] = YSm_mat
  }
  YSm.list[[sindex]] = Ysm.sublist
}



