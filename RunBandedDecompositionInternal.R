# Banded decomposition 
library(gplots)
library(latex2exp)
library(time)
# setwd("G://My Drive/BandedDecomposition") # Change to your local path
setwd("C://Code//Github//BandedDecomposition")
source("BandedDecomposition.R")


##########Plots for paper: log-scale
# Plot for Simple KacMurdockSzego Matrix:
k <- 7
n <- 2**k
D.vec <- pmin(c(0, 2**(0:(k))), n)
matrix.params <- c()
matrix.params$T <- 1 # total time 
jpeg(file="KacMurdockSzego_rho_log.jpg", width = 400, height = 300, units='mm', res = 600)
res = 100
rho.vec <- (0:(res))/(res)# c(0.1, 0.25, 0.5, 0.75, 0.9)
plot_protfolio_value('KacMurdockSzego', matrix.params, rho.vec, rep(0, n), D.vec, "by.rho", force.run = FALSE)
dev.off()


# Plot for Fractional Brownian Motion: 
jpeg(file="FracBrownMotion_H_log_n128.jpg", width = 400, height = 300, units='mm', res = 600)

#jpeg(file="FracBrownMotion_H_log_T1.jpg", width = 400, height = 300, units='mm', res = 600)
res = 100
# H.vec <- c(0.001, 1:(res-2),res)/(res)
H.vec <- c(0.001, 1:res)/(res)
plot_protfolio_value('FracBrownMotion', matrix.params, H.vec, rep(0, n), D.vec, "by.alpha", 
                     add.legend = TRUE, force.run = TRUE)
dev.off()
jpeg(file="FracBrownMotion_H_log_T2.jpg", width = 400, height = 300, units='mm', res = 600)
matrix.params$T = 2
plot_protfolio_value('FracBrownMotion', matrix.params, H.vec, rep(0, n), D.vec, "by.alpha", 
                     add.legend = TRUE, force.run = TRUE)
dev.off()
jpeg(file="FracBrownMotion_H_log_T4.jpg", width = 400, height = 300, units='mm', res = 600)
matrix.params$T = 4
plot_protfolio_value('FracBrownMotion', matrix.params, H.vec, rep(0, n), D.vec, "by.alpha", 
                     add.legend = TRUE, force.run = TRUE)
dev.off()

# 64, 128
start.time <- Sys.time()
for(k in 7:7)
{
#  k <- 7
  n <- 2**k
  D.vec <- pmin(c(0, 2**(0:(k))), n)
#  D.vec = c(1, 2) * (k-5)
  res = 100
  H.vec <- c(0.001, 1:res)/(res)
  jpeg(file=paste0("FracBrownMotion_H_log_T1_n", as.character(n), ".jpg"), width = 400, height = 300, units='mm', res = 600)
  matrix.params$T = 1
  plot_protfolio_value('FracBrownMotion', matrix.params, H.vec, rep(0, n), D.vec, "by.alpha", 
                       add.legend = TRUE, force.run = TRUE)
  dev.off()
}
run.time <- difftime(Sys.time(), start.time, units = "secs")
print(run.time)


k <- 9
n <- 2**k
D.vec <- c(0, 1, 2, 4)  #   pmin(c(0, 2**(0:(k))), n)
matrix.params <- c()
matrix.params$T <- 1 # total time 
jpeg(file=paste0("FracBrownMotion_H_log_n", as.character(n), ".jpg"), width = 400, height = 300, units='mm', res = 600)
res = 100
H.vec <- c(0.001, 1:res)/(res)
plot_protfolio_value('FracBrownMotion', matrix.params, H.vec, rep(0, n), D.vec, "by.alpha", add.legend = TRUE, force.run = FALSE)
dev.off()


# Plot for Fractional Brownian Motion: 
jpeg(file="FracBrownMotion_H_log_n128.jpg", width = 400, height = 300, units='mm', res = 600)



# Try one point the effect of n 
n=64
T = 2
protfolio_value(FracBrownMotion(H=0.2, n=n, T = 1), rep(0, n), 1)
protfolio_value(FracBrownMotion(H=0.2, n=n, T = 1), rep(0, n), 1)
protfolio_value(FracBrownMotion(H=0.2, n=T*n, T = 2), rep(0, T*n), T)

D1.vec <- rep(0, 10)
Dprop.vec <- rep(0,10)
H=0.8
for(T in 1:10)
{
  print(T)
  D1.vec[T] = protfolio_value(FracBrownMotion(H=0.2, n=T*n, T = 1), rep(0, T*n), 1)
  Dprop.vec[T] = protfolio_value(FracBrownMotion(H=0.2, n=T*n, T = 1), rep(0, T*n), T)
}
jpeg(file=paste0("FracBrownMotion_val_vs_n", as.character(n), "_H", as.character(H), ".jpg"), width = 400, height = 300, units='mm', res = 600)
plot(D1.vec, ylim=c(-1,0))
points(Dprop.vec, col="red", pch=20)
legend(1, -0.5, c("D=1", "D=n/64"), 
       col=c("black", "red"),   pch = c(1,20),
       cex=1.25, box.lwd = 0, box.col = "white", bg = "white")

dev.off()

C1 = FracBrownMotion(H=0.2, n=n, T = 1)
C2 = FracBrownMotion(H=0.2, n=2*n, T = 1)  

build_covariance_matrix("FracBrownMotion", matrix.params, n)


