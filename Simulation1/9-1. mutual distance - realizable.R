##### Realizable

rm(list=ls())

a=54325347; b=4326463

### Choose

# sig2_true=20; N_vec = c(500, 1000, 2000, 5000, 10000, 20000)
sig2_true=5000; N_vec = c(2000, 5000, 10000, 20000, 50000, 1e+05)
gridno=100


### Settings

S_size=30; A_true = 100; p_true=0.9; gamma=0.99
svec=1:S_size; mu_last=0
muR_left = A_true * p_true^(svec - 1) # E(R(s,L))
muR_right = A_true * p_true^(svec + 1); muR_right[S_size] = mu_last # E(R(s,R))
var_Z_true = sig2_true / (1-gamma^2) # Assume to be known.
theta_true = c(A_true, p_true) # 실수: 원래는 p여야 하는데, 실수로 rho로 잘못 명명했다.


if(sig2_true==20){
  theta1_est <- rbind( # var=20
    c(99.454339,0.900433),
    c(100.4777777,0.8995958),
    c(99.7943347,0.9000317),
    c(99.9600061,0.8999633),
    c(100.0264739,0.8998744),
    c(99.9458154,0.9003188)
  )  
} else {
  theta1_est <- rbind( # var=5000
    c(96.6663824,0.9007102),
    c(99.2563663,0.8996292),
    c(100.3342754,0.8981637),
    c(99.1387461,0.9049824),
    c(101.6000176,0.8995298),
    c(99.4988312,0.9010372)
  )
}



### Inaccuracy

source("subcodes/1D_Realizable Objective Functions.R")

directory=paste0("results/optimal_m/realizable_var", sig2_true,"/")
allthetas = theta1_est

for(N_ind in 1:length(N_vec)){
  
  # N_ind=1
  
  filename=paste0(directory, "EBRM_realizable_Avalue_var", sig2_true,"N",N_vec[N_ind],"Trial1to50.txt")
  A_table=read.table(filename)
  filename=paste0(directory, "EBRM_realizable_rhovalue_var", sig2_true,"N",N_vec[N_ind],"Trial1to50.txt")
  rho_table=read.table(filename)
  
  allthetas = rbind(allthetas, cbind(c(as.matrix(A_table)), c(as.matrix(rho_table))))
}


par(mfrow=c(2,length(N_vec)/2), oma=c(0,0,2,0))

for(N_ind in 1:length(N_vec)){
  
  # N_ind=1

  filename=paste0(directory, "EBRM_realizable_Avalue_var", sig2_true,"N",N_vec[N_ind],"Trial1to50.txt")
  A_table=read.table(filename)
  filename=paste0(directory, "EBRM_realizable_rhovalue_var", sig2_true,"N",N_vec[N_ind],"Trial1to50.txt")
  rho_table=read.table(filename)
  
  plot(cbind(c(as.matrix(A_table)), c(as.matrix(rho_table))), type="n", xlab="A_estimate", ylab="rho_estimate")
  # plot(allthetas, type="n", xlab="A_estimate", ylab="rho_estimate")
  
  points(t(theta_true), col=2, lwd=5, cex=3, pch=2)
  points(t(theta1_est[N_ind,]), col=1, lwd=5, cex=3, pch=2)
  

  # Arho_subtable=cbind(A_table[,"m.2"], rho_table[,"m.2"]); points(Arho_subtable, col=1, cex=0.5)
  # Arho_subtable=cbind(A_table[,"m.3"], rho_table[,"m.3"]); points(Arho_subtable, col=2, cex=0.5)
  Arho_subtable=cbind(A_table[,"m.4"], rho_table[,"m.4"]); points(Arho_subtable, col=3, cex=1)
  # Arho_subtable=cbind(A_table[,"m.5"], rho_table[,"m.5"]); points(Arho_subtable, col=4, cex=0.5)
  # Arho_subtable=cbind(A_table[,"m.6"], rho_table[,"m.6"]); points(Arho_subtable, col=5, cex=0.5)
  Arho_subtable=cbind(A_table[,"m.7"], rho_table[,"m.7"]); points(Arho_subtable, col=6, cex=1)
  # Arho_subtable=cbind(A_table[,"m.8"], rho_table[,"m.8"]); points(Arho_subtable, col=7, cex=0.5)
  # Arho_subtable=cbind(A_table[,"m.9"], rho_table[,"m.9"]); points(Arho_subtable, col=8, cex=0.5)
  Arho_subtable=cbind(A_table[,"m.10"], rho_table[,"m.10"]); points(Arho_subtable, col=9, cex=1)
  

  means=rbind(
    theta1_est[N_ind,],
    apply(cbind(A_table[,"m.2"], rho_table[,"m.2"]),2,mean),
    apply(cbind(A_table[,"m.3"], rho_table[,"m.3"]),2,mean),
    apply(cbind(A_table[,"m.4"], rho_table[,"m.4"]),2,mean),
    apply(cbind(A_table[,"m.5"], rho_table[,"m.5"]),2,mean),
    apply(cbind(A_table[,"m.6"], rho_table[,"m.6"]),2,mean),
    apply(cbind(A_table[,"m.7"], rho_table[,"m.7"]),2,mean),
    apply(cbind(A_table[,"m.8"], rho_table[,"m.8"]),2,mean),
    apply(cbind(A_table[,"m.9"], rho_table[,"m.9"]),2,mean),
    apply(cbind(A_table[,"m.10"], rho_table[,"m.10"]),2,mean)
  )
  lines(means, type="b", lwd=3)
  
  mtext( paste0("N=", N_vec[N_ind]) , cex=1, line=0.5 )
}
mtext("Parameter Flow", outer=T, cex=1.5)





Disparity_combination_list = vector("list",length(N_vec))
names(Disparity_combination_list) = paste0("N=",N_vec)

# directory=paste0("results/optimal_m/realizable_var", sig2_true,"/")

for(N_ind in 1:length(N_vec)){

  # N_ind=1

  N=N_vec[N_ind]
  set.seed(a+b*1) # We limited our experiments with seed=1.
  source("subcodes/1D Data Generation.R")

  filename=paste0(directory, "EBRM_realizable_Avalue_var", sig2_true,"N",N_vec[N_ind],"Trial1to50.txt")
  A_table=read.table(filename)
  filename=paste0(directory, "EBRM_realizable_rhovalue_var", sig2_true,"N",N_vec[N_ind],"Trial1to50.txt")
  rho_table=read.table(filename)

  steplevels <- as.numeric(gsub("m\\.", "", colnames(A_table)))

  Disparity_combination = matrix(NA, length(steplevels), 25)
  for(stepindex in 1:length(steplevels)){

    # stepindex=1

    matrix_fixedN_fixedm=cbind(A=A_table[,stepindex], rho=rho_table[,stepindex])

    index=0
    for(i in 1:25){
      index=index+1
      Disparity_combination[stepindex, index]=Disparity_thetas(matrix_fixedN_fixedm[i,], matrix_fixedN_fixedm[i+25,])
      print(paste0(index , " / 25 completed."))
    }
    print(paste0("----------------N=",N_vec[N_ind], ": step=", steplevels[stepindex], " completed."))
  }
  Disparity_combination_list[[N_ind]] = Disparity_combination
}

yrange=range(do.call(rbind, Disparity_combination_list))



par(mfrow=c(2,length(N_vec)/2), oma=c(0,0,2,0))

for(N_ind in 1:length(N_vec)){
  # N_ind=1
  matplot(steplevels, Disparity_combination_list[[N_ind]], ylab="mutual disparity")
  # matplot(steplevels, Disparity_combination_list[[N_ind]], ylab="mutual disparity", ylim=yrange)
  lines(steplevels, apply(Disparity_combination_list[[N_ind]], 1, mean), type="b")
  mtext(paste0("N=", N_vec[N_ind]), line=0.5)
}
mtext("Energy-distance between two bootstrapped estimators", outer=T, cex=1.5)

















