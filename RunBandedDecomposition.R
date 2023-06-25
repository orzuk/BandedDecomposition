# Banded decomposition 
library(gplots)
library(latex2exp)
library(time)
# setwd("G://My Drive/BandedDecomposition") 
setwd("C://Code//Github//BandedDecomposition") # Change to your local path
source("BandedDecomposition.R")


##########Plots for paper: log-scale
# Plot for Simple KacMurdockSzego Matrix:
k <- 6
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
jpeg(file="FracBrownMotion_H_log.jpg", width = 400, height = 300, units='mm', res = 600)
res = 100
H.vec <- c(0.001, 1:res)/(res)
plot_protfolio_value('FracBrownMotion', matrix.params, H.vec, rep(0, n), D.vec, "by.alpha", 
                     add.legend = TRUE, force.run = TRUE)
dev.off()



# Some specific values for large n to see the effect of n 
n=64 # First keep D fixed
protfolio_value(FracBrownMotion(H=0.2, n=n), rep(0, n), 1)
protfolio_value(FracBrownMotion(H=0.2, n=2*n), rep(0, 2*n), 1)
protfolio_value(FracBrownMotion(H=0.2, n=4*n), rep(0, 4*n), 1)

# Next keep D_n/n fixed:
protfolio_value(FracBrownMotion(H=0.2, n=2*n), rep(0, 2*n), 2)
protfolio_value(FracBrownMotion(H=0.2, n=4*n), rep(0, 4*n), 4)
protfolio_value(FracBrownMotion(H=0.2, n=16*n), rep(0, 16*n), 16)