# Banded decomposition figures 
library(gplots)
library(latex2exp)
library(Matrix)
library(ltsa)
library(rlang)
# Change path to project's directory 
setwd("C://Code//Github//BandedDecomposition") # Change to your local path

source("BandedDecomposition.R")
source("SemiStatic.R")

################################################################
# FIGURE 1 : Compute and plot strategy 
d=0.2
sigma2 = 0.5
a.vec = plot_gamma_weights(sigma2, d, 
                           fig.file = paste0("figs/kappa_H_", as.character(d), "_sigma2_", as.character(sigma2), ".jpg"))
sigma2 = 2
a.vec = plot_gamma_weights(sigma2, d, 
                           fig.file = paste0("figs/kappa_H_", as.character(d), "_sigma2_", as.character(sigma2), ".jpg"))
################################################################


################################################################
# FIGURE 2 : Now plot the value of the proportional model as a function of
# 0 < d < 1 and sigma^2
jpeg(file=paste0("val_prop_limit.jpg"), 
     width = 400, height = 300, units='mm', res = 600)
d.res <- seq(0.005, 0.995, 0.005)
num.d <- length(d.res)
sigma.res <- exp(seq(-3, 3, 0.01)) # values of sigma2
num.sigma <- length(sigma.res)

z <- matrix(0, nrow=num.sigma, ncol=num.d)
for(i in 1:num.sigma)
  for(j in 1:num.d)
    z[i,j] = option_invest_value_prop_lim(sigma.res[i], d.res[j])    
z = -exp(z)

colMap <- colorRampPalette(c("red","white","blue" ))(num.sigma*num.d)
par(mar=c(8,6,4,1)+.1)
image(log(sigma.res), d.res, z, col = colMap, ylab="H", xlab="", # xlab=TeX("$log(sigma^2)$"), 
      main="", cex=3, cex.lab=3, cex.axis=3, cex.main=3)
title(xlab=TeX("$log(\\frac{\\hat{varsigma}^2}{varsigma^2})$"), line=7, cex.lab=3, family="Calibri Light")

show.vals = as.character(round(seq(min(z, na.rm=TRUE), max(z, na.rm=TRUE), length.out=10), 2))
show.inds = round(seq(1, length(colMap), length.out=10))
legend(grconvertX(8200, "device"), grconvertY(550, "device"),
       show.vals, fill = colMap[show.inds], xpd = NA, cex=2)
dev.off()
############### Plot the same as 3D surface ####################
library(plot3D)
#surf3D(x, y, z, colvar = z, colkey = TRUE, 
#       box = TRUE, bty = "b", phi = 20, theta = 120)
Meshi <- mesh(d.res, sigma.res)
u <- Meshi$x ; v <- Meshi$y
x <- u # v * cos(u)
y <- log(v) # * sin(u)
library(plotly)
fig <- plot_ly(x = x, y = y, z = t(z)) %>% add_surface() %>%
  layout(scene = list(xaxis = list(title = "H"), yaxis = list(title = ""), zaxis = list(title="U"))) %>% 
  config(mathjax = 'cdn')
fig  # need to save manually 


################################################################
################################################################

# Revision figures here: 
# Figure 3: 
h.vec=c(0.01, 0.05, 0.1, 0.2, 0.5)
sigma2.vec = c(0.2, 0.25, 0.5, 0.9, 1.1, 2, 4, 5)

for(sigma2 in sigma2.vec)
{
  aa = plot_kappa_limit_weights(sigma2, (h.vec), 
                             fig.file = paste0("figs/kappa_vs_H_limit_sigma2_", as.character(sigma2), ".jpg"))
  kk = plot_kappa_limit_weights(sigma2, (h.vec), 
                                fig.file = paste0("figs/kappa_vs_H_limit_sigma2_", as.character(sigma2), "_no_diff.jpg"), FALSE)
}

sigma2.vec = c(0.2, 0.5, 0.9, 1.1, 2, 5)
plot_kappa0_limit(sigma2.vec, fig.file = paste0("figs/kappa0_vs_H_limit.jpg"))
plot_kappa0_limit(sigma2.vec, fig.file = paste0("figs/kappa0_vs_H_limit.jpg"), log.flag = TRUE)
plot_kappa0_limit(sigma2.vec, fig.file = paste0("figs/alphaH_vs_H_limit.jpg"), plot.alphaH = TRUE)


################################################################

### Old/unused figures below

################################################################
# FIGURE: Compute and plot log determinant normalized as a function of n 
# using both Exact computation and asymptotic limit  
sigma2 = 2
D.vec = c(1,2,4,8,16)
num.D <- length(D.vec)
col.vec <- get_color_vec("red", "blue", num.D)
par(mar=c(5,6,4,1)+.1)
n.vec = unique(round(2**(seq(1, 10, 0.2)))) # 2:200
n.vec = n.vec[n.vec > max(D.vec)]

#plot(scalar.params.vec, value.mat[,1], xlab=TeX(paste0("$", params.str[matrix.type], "$")), #    xlab=expression(params.str[matrix.type]), 
#     ylab="Value", col=col.vec[1], ylim=range(value.mat), type="l", lwd=2, 
#     cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)

for(sigma2 in c(2, 0.5))
{
  if(sigma2==2)
  {
    y.lim <- c(-0.37, 0)
    y.leg <- -0.29
  } else # 0.5
  {
    y.lim <- c(0, 0.32)
    y.leg <- 0.31
  }
  jpeg(file=paste0("LogDetQ_sigma2_", as.character(sigma2), ".jpg"), 
       width = 400, height = 300, units='mm', res = 600)
  par(mar=c(5,6,4,1)+.1)
  for(i in 1:num.D)
  {
    D = D.vec[i]
    print(paste0("D=", as.character(D)))
    
    exact <- Q_log_det(sigma2, n.vec, D)
    alim <- sigma2_to_a(sigma2, -1, D)
    #    detQlim <- log(4) + 2*log(D+1) - log(sigma2) - 2*log( sqrt(1+4*D*(D+1)/sigma2) +1  )
    detQlim <- 2*log(2*(D+1) / (sqrt(sigma2) + sqrt(sigma2+4*D*(D+1)))   )
    
    if(i==1)
      plot(log10(n.vec), exact$logdetQ.vec, pch=20, cex=1.5, col=col.vec[i], 
           cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, 
           xlab=TeX("$log_{10}(n)$"), ylab=TeX("$log(|Q|)/n$"), 
           ylim = y.lim) # , main="log(|Q|)/n vs. n: exact vs limit")
    else
      points(log10(n.vec), exact$logdetQ.vec, pch=20, cex=1.5, col=col.vec[i], 
             cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2) # , main="log(|Q|)/n vs. n: exact vs limit")
    abline(a=detQlim, b=0, col=col.vec[i], lwd=1.5)
  }
  add.legend = 1
  if(add.legend)
    legend(1.2, y.leg, paste0(rep("D=", num.D), as.character(D.vec)),
           col=col.vec[1:num.D],  lty=rep(1, num.D), lwd=2,
           cex=1.9, box.lwd = 0, box.col = "white", bg = "white")
  grid(col = "darkgray", lwd=1.5)
  dev.off()
} # end loop on sigma2
################################################################


################################################################
# Figure: Plot curves of the limit as function of sigma2, 
# for D=1,2,..
jpeg(file=paste0("LogDetQ_limit.jpg"), 
     width = 400, height = 300, units='mm', res = 600)
par(mar=c(5,6,4,1)+.1)
sigma.vec <- exp(seq(-5, 5, 0.1)) # values of sigma2
y.lim <- c(-4,1)
for(i in 1:num.D)
{
  D = D.vec[i]
  DetQlim.vec <- log(4) + 2*log(D+1) - 
    log(sigma.vec) - 2*log( sqrt(1+4*D*(D+1)/sigma.vec) +1  )
  if(i==1)
    plot(log(sigma.vec), DetQlim.vec, type="l", lwd=1.5, col=col.vec[i], 
         cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, 
         xlab=TeX("$log(sigma^2)$"), ylab = TeX("$log(|Q|)/n$"), 
         ylim = y.lim) # , main="log(|Q|)/n vs. n: exact vs limit")
  else
    lines(log(sigma.vec), DetQlim.vec, type="l", lwd=1.5, col=col.vec[i], 
          cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2) # , main="log(|Q|)/n vs. n: exact vs limit")
}
legend(-4.8, -2.8, paste0(rep("D=", num.D), as.character(D.vec)),
       col=col.vec[1:num.D],  lty=rep(1, num.D), lwd=2,
       cex=1.9, box.lwd = 0, box.col = "white", bg = "white")
grid(col = "darkgray", lwd=1.5)
dev.off()
################################################################
#### Now the same but with the investor's value 
jpeg(file=paste0("val_limit.jpg"), 
     width = 400, height = 300, units='mm', res = 600)
par(mar=c(5,6,4,1)+.1)
sigma.vec <- exp(seq(-5, 5, 0.1)) # values of sigma2
num.sigma <- length(sigma.vec)
y.lim <- c(-0.5,10)
for(i in 1:num.D)
{                                                                                                                                                                                                                                                                                                                                                                                   D = D.vec[i]
  vallim.vec <- rep(0, num.sigma)
  for(j in 1:num.sigma)    
    vallim.vec[j] = option_invest_value(sigma.vec[j], -1, D)$val

  if(i==1)
    plot(log(sigma.vec), vallim.vec, type="l", lwd=1.5, col=col.vec[i], 
       cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, 
       xlab=TeX("$log(sigma^2)$"), ylab = TeX("$V(D, sigma^2)$"), 
       ylim = y.lim) # , main="log(|Q|)/n vs. n: exact vs limit")
  else
    lines(log(sigma.vec), vallim.vec, type="l", lwd=1.5, col=col.vec[i], 
        cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2) # , main="log(|Q|)/n vs. n: exact vs limit")
}
legend(-5, 10, paste0(rep("D=", num.D), as.character(D.vec)),
       col=col.vec[1:num.D],  lty=rep(1, num.D), lwd=2,
       cex=1.9, box.lwd = 0, box.col = "white", bg = "white")
grid(col = "darkgray", lwd=1.5)
dev.off()



################################################################
# Now plot the a of the proportional model as a function of
# 0 < d < 1 and sigma^2
jpeg(file=paste0("a_prop_limit.jpg"), 
     width = 400, height = 300, units='mm', res = 600)
d.res <- seq(0.01, 0.99, 0.01)
num.d <- length(d.res)
sigma.res <- exp(seq(-3, 3, 0.02)) # values of sigma2
num.sigma <- length(sigma.res)

A <- matrix(0, nrow=num.sigma, ncol=num.d)
for(i in 1:num.sigma)
  for(j in 1:num.d)
  {
    A[i,j] = sigma2_to_a_limit(sigma.res[i], d.res[j], prop.flag = TRUE)    
    A[i,j] = A[i,j] / (1-A[i,j]*d.res[j])
  }
#z = log(0.0000000000001+z)
# z = log(1+z)

colMap <- colorRampPalette(c("red","white","blue" ))(num.sigma*num.d)
image(log(sigma.res), d.res, A, col = colMap, ylab="d", xlab=TeX("$log(sigma^2)$"), 
      main="log(1+v)")

show.vals = as.character(round(seq(min(A, na.rm=TRUE), max(A, na.rm=TRUE), length.out=10), 2))
show.inds = round(seq(1, length(colMap), length.out=10))
legend(grconvertX(0.5, "device"), grconvertY(1, "device"),
       show.vals, fill = colMap[show.inds], xpd = NA)
dev.off()



################################################################
# Show convergence to value for log-det 
d.res = c(0.1,0.2,0.3,0.4,0.5)
num.d = length(d.res)
for(sigma2 in c(2, 0.5))
{
  if(sigma2==2)
  {
    y.lim <- c(-5, 0)
    y.leg <- -4
  } else # 0.5
  {
    y.lim <- c(0, 5)
    y.leg <- 1
  }
  jpeg(file=paste0("LogDetQ_prop_sigma2_", as.character(sigma2), ".jpg"), 
       width = 400, height = 300, units='mm', res = 600)
  par(mar=c(5,6,4,1)+.1)
  for(i in 1:num.D)
  {
    d = d.res[i]
    print(paste0("d=", as.character(d)))
    D.vec = round(n.vec*d)
    exact <- Q_log_det(sigma2, n.vec, D.vec)
    alim <- sigma2_to_a(sigma2, -1, D.vec)
    #    detQlim <- log(4) + 2*log(D+1) - log(sigma2) - 2*log( sqrt(1+4*D*(D+1)/sigma2) +1  )
    detQlim <- 2*log(2*(D.vec+1) / (sqrt(sigma2) + sqrt(sigma2+4*D.vec*(D.vec+1)))   )
    
    
    an.lim = ( d-2*sigma2+sqrt(d*d+4*sigma2*(1-d)) ) / (2*sigma2*d)   # NEED TO CHANGE !!! constant such that a_n * n -> an.lim
    detQlimProp <- an.lim * (1-d) / (1+an.lim*d) + log(1+an.lim*d)  # here we do not divide by n !!! 
    
    if(i==1)
      plot(log10(n.vec), exact$logdetQ.vec*n.vec, pch=20, cex=1.5, col=col.vec[i], 
           cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2, 
           xlab=TeX("$log_{10}(n)$"), ylab=TeX("$log(|Q_n|)$"), 
           ylim = y.lim) # , main="log(|Q|)/n vs. n: exact vs limit")
    else
      points(log10(n.vec), exact$logdetQ.vec*n.vec, pch=20, cex=1.5, col=col.vec[i], 
             cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2) # , main="log(|Q|)/n vs. n: exact vs limit")
    abline(a=detQlimProp, b=0, col=col.vec[i], lwd=1.5)
  }
  add.legend = 1
  if(add.legend)
    legend(1.2, y.leg, paste0(rep("d=", num.d), as.character(d.res)),
           col=col.vec[1:num.d],  lty=rep(1, num.d), lwd=2,
           cex=1.9, box.lwd = 0, box.col = "white", bg = "white")
  grid(col = "darkgray", lwd=1.5)
  dev.off()
} # end loop on sigma2







################################################################
################################################################
################################################################
################################################################
### Old and temp stuff below
################################################################
################################################################
################################################################
################################################################



# Compute via fourier coefficients: 
max.k=10 # number of Fourier coefficients to take in the expansion
blim <- const_plus_I_to_Q_recursive(alim, max.k, D) # compute large matrix 

limit = integrate(finite_fourier, 0, 1, c=blim$t.vec)


# Plot a the root a
plot(n.vec, exact$a.vec, pch=20, xlab="n", ylab="a_n", main="a_n: exact vs. limit (red)")
abline(a=alim, b=0, col="red")

# Plot as function of n: 
sigma2 = 1
c <- c(1:5)
#c <- c(2,1,2,1,2)
c = c*sigma2/sum(c) # Normalize 
n.vec <- c((length(c)+1):200)
num.n <- length(n.vec)
ld <- rep(0, num.n)

for(i in 1:num.n)
  ld[i] = banded_toeplitz_to_log_det(c, n.vec[i])$logdet/n.vec[i]


plot(n.vec, ld, xlab="n", ylab="log(|Q_n|)/n")  
print(ld[num.n])  # take limit




########################################################################
## Here try banded decomposition of special matrix                    #  
########################################################################
n = 7
a = 2
D = 2
n = 4
A = diag(rep(1, n)) + matrix(a, n, n)
BD4 = banded_decomposition(A, D)
n = 5
A = diag(rep(1, n)) + matrix(a, n, n)
BD5 = banded_decomposition(A, D)



for(D in c(1,2))
{
  max.n = 10
  gamma.vec = matrix(0, nrow = max.n, ncol = max.n)
  qinv.vec = matrix(0, nrow = max.n, ncol = max.n)
  for(n in c((D+1):max.n))
  {
    A = diag(rep(1, n)) + matrix(a, n, n)
    BD = banded_decomposition(A, D)
    gamma.vec[n,1:n] = BD$Gamma[1, ]
    qinv.vec[n,1:n] = BD$QInv[1,]  
  }
  print(paste0("D=", as.character(D)))
  print("Q^-1")
  print(round(qinv.vec[(D+1):max.n,], 5))
  print("Gamma")
  print(round(gamma.vec[(D+1):max.n,(D+1):max.n], 5))
}


print(BD$QInv)
is_toeplitz(BD$QInv, epsilon=0.0000000001)
round(solve(BD$QInv), digits=10)
is_toeplitz(solve(BD$QInv))

# Try decomposition
n = 15
a = 2
D = 5
A = diag(rep(1, n)) + matrix(a, n, n)

b = const_plus_I_to_Q_recursive(a,n,D)

det(b$QInv)
dd = rep(1, n-D)
DD = rep(1, n-D)
for(i in 1:(n-D))
{
  DD[i] = det(b$QInv[i:(i+D),i:(i+D)])
}
if(n-D>=2)
  for(i in 2:(n-D))
  {
    dd[i] = det(as.matrix(b$QInv[i:(i+D-1),i:(i+D-1)]))
  }
print(det(b$QInv))
print(prod(DD)/prod(dd))
print(  (1+(D+1)*a)**(n-D) / (1+D*a)**(n-D-1)   )

print(DD)
print(dd[2:(n-D)])


QInv = toeplitz(b)
Gamma = A - QInv
Q = solve(QInv)

dev.off()
heatmap.2((abs(Q)>0.0000005)+0.0000001, Rowv=FALSE, Colv=FALSE)
heatmap.2(Q, Rowv=FALSE, Colv=FALSE)
heatmap.2(QInv, Rowv=FALSE, Colv=FALSE)




# x = (1+a)/a








# [1, a]  is posdef for a>-1/(n-1)
# [1+a, a] is posdef for a/(1+a) > -1/(n-1)
# a > (1+a) / (n-1)
# a * (n-2) / (n-1) > 1 / (n-1)
# a > 1 / (n-2)




n=5
a = -1/(n-1)-0.0000000001
B = diag(rep(1-a, n)) + matrix(a, n, n)
isposdef(B)

a = -1/(n)-0.0000000001
A = diag(rep(1, n)) + matrix(a, n, n)
isposdef(A)

n = 3
a = 1
D = 2
b3 = const_plus_I_to_Q_recursive(a,3,D)
b4 = const_plus_I_to_Q_recursive(a,4,D)


b9 = const_plus_I_to_Q_recursive(a,9,D)
b10 = const_plus_I_to_Q_recursive(a,10,D)


max(abs(bb$QInv %*% bb$Q - diag(n)))
bb$Q.sum - sum(bb$Q)

ccc = const_plus_I_to_Q_recursive(a,n,D)

n=100
D=2
sigma2=0.5

sigma2.vec = exp(seq(-5, 5, 0.01))
n.sig = length(sigma2.vec)
aa <- bb <- rep(0, n.sig)

for(i in 1:n.sig)
{
  aa[i] = sigma2_to_a(sigma2.vec[i], n, D)[1]
  bb[i] = sigma2_to_a(sigma2.vec[i], n, D)[2]
}
plot(sigma2.vec, aa, type="l")
lines(sigma2.vec, bb, col="red")
which(bb > -1/n)

# Check consistency
bb = const_plus_I_to_Q_recursive(aa[1],n,D)
bb$Q.sum - n*sigma2
bb = const_plus_I_to_Q_recursive(aa[2],n,D)
bb$Q.sum - n*sigma2

diag(Q)
round(Q, 6)
dev.off()
heatmap.2((abs(Q)>0.0000005)+0.0000001, Rowv=FALSE, Colv=FALSE)


# Formula for a: 
# We get that 
# (aD+1)(aD+1+a) n\sigma^2 n+aD(D+1)

# a_{1,2} = \frac{D-n (2D+1)\sigma^2 \pm \sqrt{(n\sigma^2 (2D-1)-D)^2 - 
# 4n^2\sigma^2 (\sigma^2-1)(D^2-D) )}}{2n \sigma^2(D^2+D)}

# (ns^2(2D+1)-D)

# Check new formula for the matrix Q^-1
a = 1.5 
n = 5
D = 2
