# Banded decomposition 
library(gplots)
library(latex2exp)
library(Matrix)
library(ltsa)
# Change path to project's directory 
setwd("C://Code//Github//BandedDecomposition") # Change to your local path

source("BandedDecomposition.R")


# Off-Diagonal 1 -> a
# Diagonal x -> 1 + a
# find
# Input: 
# a - constant
# n - matrix size 
# D - delay 
# Diagonal 1+a, off-diagonal a
# Output: 
# Q - Symmetric D-banded matrix 
# QInv - Symmetric Toeplitz matrix such that Q*QInv = I_n
# t.vec - vector for Toeplitz matrix QInv such that QInv_{ij} = t.vec_{|i-j|}
const_plus_I_to_Q_recursive <- function(a, n, D)
{
  # Matrix Q^-1 (Toeplitz)
  t.vec = c(1+a, rep(a, D), rep(0, n-D-1))
  if(D+1<n)
    for(i in min(D+2):n)
      t.vec[i] = sum(t.vec[(i-D):(i-1)]) /  (D+1/a)    
  QInv = toeplitz(t.vec)
  
  # Matrix Q (D-banded)
  Q = matrix(0, nrow=n, ncol=n)
  Q[1,1] <- (a*D+1)/(a*D+1+a)
  Q[2:(D+1),1] <- Q[1,2:(D+1)] <- Q[1,1]-1
  for(i in 1:(n-1))
    for(j in 1:(n-1)) #     for(j in max(1,i-D):min(n-1,i+D))
      Q[i+1,j+1] = Q[i,j] + ((max(i,j)<=D)-(min(i,j)>=(n-D))) *  a**2/((a*D+1)*(a*D+1+a))
  Q.sum = (n+a*D*(D+1)) / ((a*D+1)*(a*D+1+a))
  
  
  # New: Compute Q (a D-banded matrix, the inverse of QInv) using a direct, non-recursive formula
  v.vec = c((a*D+1)/(a*D+1+a), rep(-a/(a*D+1+a), D), rep(0, n-D-1))
  Q2 = matrix(0, nrow=n, ncol=n)
  for(i in c(1:n))
    for(j in c(1:n))
    {
      for(k in c(1:min(i,j)))
        Q2[i,j] <- Q2[i,j] + v.vec[i-k+1]*v.vec[j-k+1]
      if(min(i,j)>1)
        for(k in c(1:(min(i,j)-1)))
          Q2[i,j] <- Q2[i,j] - v.vec[n-i+k+1]*v.vec[n-j+k+1]
    }
  Q2 = Q2 / v.vec[1]
  
  # Alternative computation: 
#  T = toeplitz(v.vec)
#  TL = T
#  TL[upper.tri(TL)] = 0
#  SL = matrix(0, nrow=n, ncol=n)
#  for(i in 2:n)
#    for(j in 1:(i-1))
#      SL[i,j] = v.vec[n-i+j+1]
#  Q3 = (TL %*% t(TL) - SL %*% t(SL)) / v.vec[1]
  
  return(list(QInv=QInv, Q=Q, t.vec=t.vec, Q.sum=Q.sum, Q2=Q2))  
}

# Solve Qudaratic equation to get the value a 
# Input: 
# sigma2 - investor's variance
# n - number of steps
# D - delay 
# Output: 
# a - constant such that the matrix I_n + a 1^n satisfies ... 
# (return both roots, take first)
sigma2_to_a <- function(sigma2, n, D)
{
  if(n < 0) # limit n-> infinity 
    return( ( sqrt(1+4*D*(D+1)/sigma2) - 2*D-1 ) / (2*D*(D+1))      )
  a = n*sigma2*D*(D+1)
  b = n*sigma2*(2*D+1)-D*(D+1)
  c = n*(sigma2-1)
  Delta = sqrt(b**2 - 4*a*c)
  
  return( c((-b + Delta)/(2*a), (-b - Delta)/(2*a))  )
}



#detQlim <- 2*log(2*(D+1) / (sqrt(sigma2) + sqrt(sigma2+4*D*(D+1)))   )


# lim n-> infty of value when we take D_n = d*n, and have a fixed 0<d<1
# 
option_invest_value_prop_lim <- function(sigma2, d)
{
  an.lim = ( d-2*sigma2+sqrt(d*d+4*sigma2*(1-d)) ) / (2*sigma2*d)   # NEED TO CHANGE !!! constant such that a_n * n -> an.lim
  Qlog.det.lim <- an.lim * (1-d) / (1+an.lim*d) + log(1+an.lim*d)  # here we do not divide by n !!! 
  return( 0.5*(Qlog.det.lim - an.lim * sigma2)) # value of investment 
}

n=20

# Should be positive: 
(1+(n*d+1)*an.lim/n)**(n*(1-d)) / (1+(n*d)*an.lim/n)**(n*(1-d)-1)
(1+(n*d+1)*an.lim/n)**(n*(1-d)) / (1+(n*d)*an.lim/n)**(n*(1-d))


(1 + an.lim / ((1+an.lim*d)*n))**(n*(1-d))



# Get the value at the limit
option_invest_value <- function(sigma2, n, D)
{
  if(n < 0)  # limit 
  {
    a = sigma2_to_a(sigma2, n, D)
    Qlog.det <- Q_log_det(sigma2, n, D)$logdetQ.vec
    val <- -0.5*(a*sigma2 - Qlog.det)
  } else
  {
    val = c()
    Qlog.det = c()
  }
  return(list(Qlog.det=Qlog.det, val=val))
}


# Compute the log-determinant of an n*n D-banded Toeplitz matrix
# Input: 
# A - square n*n matrix
# epsilon - tolerance
#
banded_toeplitz_to_log_det <- function(c, n)
{
  return( logdet(toeplitz(  c(c, rep(0, n-length(c))) ))     )
}

# Compute the normalized log-determinant for different values of n.
# Should we also change a? probably yes !!! 
#
Q_log_det <- function(sigma2, n.vec, D.vec)
{
  num.n <- length(n.vec)
  max.n <- max(n.vec)
  logdetQ.vec <- rep(0, num.n)
  a.vec <-  rep(0, num.n)
  for(i in 1:num.n)
  {
    if(length(D.vec)==1)
      D=D.vec
    else
      D=D.vec[i]
    a.vec[i] <- sigma2_to_a(sigma2, n.vec[i], D)[1]
    if(n.vec[i] < 0) # Give limit as n->infinity
      logdetQ.vec <- 2*log(2*(D+1) / (sqrt(sigma2) + sqrt(sigma2+4*D*(D+1)))   )
    else
    {
      b <- const_plus_I_to_Q_recursive(a.vec[i], n.vec[i], D) # compute large matrix 
      logdetQ.vec[i] <- logdet(b$QInv)$logdet / n.vec[i]
    }
    #    print(a.vec[i])
#    print(logdetQ.vec[i])
  }
  return(list(logdetQ.vec=logdetQ.vec, a.vec=a.vec))
}

# Alternative: Use Szego's Theorem. Take the negative ones too? (symmetric!)
finite_fourier <- function(x, c)
{
  s <- rep(0, length(x))
  for(k in 1:length(c))
    s <- s + cos(k*x)  #      exp(2*pi*1i*k*x)
  return(  Re(log(s)))
}



# New: compute and plot the actual investment strategy in the proportional model
# as a function of the proportional delay d and the variance sigma2
gamma_weights <- function(sigma2, d, n)
{
  D = round(n*d)
  # first set vec
  a <- sigma2_to_a(sigma2, n, D)[1]

  # Vector of weights 
  gamma.vec = c(rep(a, D), rep(0, n-D))
  for(i in min(D+1, n):n)
      gamma.vec[i] = sum(gamma.vec[(i-D):(i-1)]) /  (D+1/a)    
  gamma.vec <- gamma.vec - a # no cumsum ! 
#  gamma.vec <- cumsum(gamma.vec - a)
  return(gamma.vec)
}
  
plot_gamma_weights <- function(sigma2, d, n.vec = c(100, 1000, 10000, 100000), fig.file = c())
{
  
  if(!is_empty(fig.file))
  {
    jpeg(file=fig.file, 
         width = 400, height = 300, units='mm', res = 600)
    par(mar=c(5,6,4,1)+.1)
  }
  num.n <- length(n.vec)
  a.vec <- rep(0, num.n)
  col.vec = get_color_vec(num.c=length(n.vec))
  for(i in 1:length(n.vec))
  {
    a.vec[i] <- sigma2_to_a(sigma2, n.vec[i], round(n.vec[i]*d))[1]
    gamma.vec <- gamma_weights(sigma2, d, n.vec[i])
    t.vec <- (1:n.vec[i]) / n.vec[i]
    if(i == 1)
    {
      y.lim <- range(gamma.vec*(n.vec[i]))*1.1
      plot(t.vec, gamma.vec*(n.vec[i]), type="l", lwd=2.5, col=col.vec[i], ylim = y.lim, 
           xlab="t", ylab=TeX("$b(t)$"), cex=3, cex.axis=3, cex.lab=3, cex.main=3, 
           main=TeX(paste0("Strategy: $sigma^2=", as.character(sigma2), " ; d=", as.character(d), "$")))
    } else
      lines(t.vec, gamma.vec*(n.vec[i]), lwd=2, col=col.vec[i]) # how to normalize? by n.vec[i]?
  }
  add.legend = 1
  if(add.legend)
    legend(0.7, y.lim[2], paste0(rep("n=", length(n.vec)), as.character(n.vec)),
           col=col.vec,  lty=rep(1, length(n.vec)), lwd=2.5,
           cex=3, box.lwd = 0, box.col = "white", bg = "white")
  grid(col = "darkgray", lwd=1.5)
  if(!is_empty(fig.file))
    dev.off()
  
  return(a.vec)
}  


################################################################
########### End Code of Functions ##############################
################################################################
########### Start Running Functions ############################
################################################################
