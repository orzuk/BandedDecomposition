# Banded decomposition 
library(gplots)
library(latex2exp)
setwd("G://My Drive/BandedDecomposition") # Change to your local path
source("BandedDecomposition.R")


##########Plots for paper: log-scale
# Plot for Simple Toeplitz
k <- 6
n <- 2**k
D.vec <- pmin(c(0, 2**(0:(k))), n)
jpeg(file="KacMurdockSzego_rho_log.jpg", width = 400, height = 300, units='mm', res = 600)
res = 100
rho.vec <- (0:(res))/(res)# c(0.1, 0.25, 0.5, 0.75, 0.9)
plot_protfolio_value('KacMurdockSzego', rho.vec, rep(0, n), D.vec, "by.rho", force.run = FALSE)
dev.off()


jpeg(file="FracBrownMotion_H_log.jpg", width = 400, height = 300, units='mm', res = 600)
res = 100
H.vec <- c(0.001, 1:(res-2),res)/(res)
plot_protfolio_value('FracBrownMotion', H.vec, rep(0, n), D.vec, "by.alpha", 
                     add.legend = TRUE, force.run = FALSE)
dev.off()