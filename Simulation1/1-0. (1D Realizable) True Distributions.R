rm(list=ls())
# dev.off()

##### Parameter Setting

### Environment
# gamma = 0.50
gamma = 0.99
mu_last = 0
gridno = 100


### Setting
S_size=30; A_true = 100; sig2_true = 20; p_true=0.9 # setting2: reasonable setting
# S_size=30; A_true = 100; sig2_true = 5000; p_true=0.9 # setting2: reasonable setting


##### Plot the True Distribution

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

plot(svec, muZ_left, ylim=c(ymin, ymax), main=expression(paste("True return distribution Z", pi, "(s,a)")), xlab="state", ylab="value", type="n")
# plot(svec, muZ_left, ylim=c(ymin, ymax), main="True Return Distributions", xlab="state", ylab="value")
for (sindex in svec) {
  # sindex=1
  lines(c(sindex, sindex), c(left_upper[sindex], left_below[sindex]), col="red", lwd=3)
}
lines(cbind(svec, muZ_left), col="red", lwd=3, lty="dotted")
points(svec, muZ_left, ylim=c(ymin, ymax), pch=16, cex=1.2, col="red")
for (sindex in svec) {
  # sindex=1
  lines(c(sindex+0.05, sindex+0.05), c(right_upper[sindex], right_below[sindex]), col="blue", lwd=3)
}
lines(cbind(svec, muZ_right), col="blue", lwd=3, lty="dotted")
points(svec, muZ_right, ylim=c(ymin, ymax), pch=16, cex=1.2, col="blue")

legends = c("a=(-1)", "a=   1")
legend("topright", legend = legends, col=c("red", "blue"), lty=1)


mtext(paste0("variance=", sig2_true, " & gamma=", gamma), line=0.5)











