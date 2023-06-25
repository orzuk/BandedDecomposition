# Banded decomposition 
library(gplots)
library(latex2exp)
library(Matrix)
library(ltsa)

# Change path to project's directory 
# setwd("G://My Drive/BandedDecomposition") 

# Check if a matrix is Toeplitz
# Input: 
# A - square n*n matrix
# epsilon - tolerance
#
is_toeplitz <- function(A, epsilon=0)
{
  n <- dim(A)[1]
  for(i in 1:(n-1))
  {
    diag.range <- range(diag(A[i:n,]))
    if(diag.range[2] > diag.range[1]+epsilon)
      return(FALSE)
    diag.range <- range(diag(A[,i:n]))
    if(diag.range[2] > diag.range[1]+epsilon)
      return(FALSE)
  }
  return(TRUE)
}

# return the log of the determinant (useful to avoid over/under-flows for small/large matrices)
# For negative matrices 
logdet <- function(A)
{
#  return( sum(log(eigen(A)$values)) )  # use eigenvalues
#  mu = max(A)
#  log(det(A / mu)) + 
  mu =  max(abs(A)) 
  d = det(A / max(abs(A)))
  
  return( list(logdet=log(abs(d)) + dim(A)[1]*log(mu), signdet=sign(d)  ))
}
  


# Decompose A^-1 as A^-1 = Q + Gamma
# Input: 
# A - a positive definite matrix n*n
# D - an integer between 1 and n (band width)
# Output: 
# Q - a matrix such that Q^{-1}_{ij}=A^{-1}_{ij} for |i-j}<=H, and that Q^{-1} is D-banded
# Gamma - = A^{-1}-Q^{-1}, an 'anti' D-banded matrix
banded_decomposition <- function(A, D) {
  n = dim(A)[1]
  if(is_toeplitz(A))  # Faster trench inverse 
    AInv = TrenchInverse(A)
  else
    AInv=solve(A)  # Take inverse 
  if(D==0)
    Q = diag(diag(AInv))
  else
  {
    Q=AInv
    if(D+1<=n-1)  # for n-1 no need to run loop (error in range if we do)
      for(m in (D+1):(n-1))
        for(i in 1:(n-m))
        {
          Q[i, i+m] <- 0
          inds <- (i+1):(i+D)
          for(j in 1:D)
            Q[i, i+m] <- Q[i, i+m] + (-1)**j * Q[i,i+j] * det( matrix(Q[inds , c(  inds[inds != i+j], i+m )], nrow=D))
          Q[i, i+m] <- (-1)**(D) * Q[i, i+m] /  det(matrix(Q[ inds , inds ], nrow=D)) # compute minor 
          Q[i+m, i] <- Q[i, i+m]  # symmetric
          
          if(is.na(Q[i, i+m]) | is.infinite(Q[i, i+m]))  # problem! underflow/overflow
          {
            logdet.vec <- rep(0, D)
            signdet.vec <- rep(0, D)
            for(j in 1:D)
            {
              ld <- logdet( matrix(Q[inds , c(  inds[inds != i+j], i+m )], nrow=D) )
              logdet.vec[j] <- ld$logdet
              signdet.vec[j] <- ld$signdet
            }
            denom <-  logdet(matrix(Q[ inds , inds ], nrow=D))
            Q[i, i+m] <- sum(Q[i, (i+1):(i+D)] * (-1)**(1:D) * signdet.vec * exp(logdet.vec - denom$logdet)) * denom$signdet * (-1)**D
            if(is.na(Q[i, i+m]))
              return(9)
            Q[i+m, i] <- Q[i, i+m]  # symmetric
          }
        }
  }
  Gamma = AInv-Q  
  return(list(QInv=Q, Gamma=Gamma)) # Name it as Q^-1
}

# Compute investor's value
# Input: 
# A - covariance matrix
# mu - mean vector
# D - delay
# Output: 
# The optimal value for an investor with increments X ~ N(mu, A) and delay D
protfolio_value <- function(A, mu, D)
{
  if(rankMatrix(A)[1] < dim(A)[1]) # not full rank (full information)
  {
    if(D < dim(A)[1]) 
      return(0) 
    else # here no information! 
    {
      print("No information for investing!")
      print(-exp(-0.5*mu %*% A %*% mu))
      return(-exp(-0.5*mu %*% A %*% mu))
    }
  }
  BD <- banded_decomposition(A, D)
  if( (det(A)==0) || is.infinite(det(BD$QInv)) ) # singularities
    return(-exp(-0.5*mu %*% A %*% mu) / sqrt(   exp( logdet(BD$QInv)$logdet+logdet(A)$logdet ) ))
  return( -exp(-0.5*mu %*% A %*% mu) / sqrt(det(BD$QInv) * det(A))   ) # no transpose of mu in R
}


# General function for generating Covariance matrices of different types
build_covariance_matrix <- function(matrix.type, matrix.params, n)
{
  if(matrix.type == "KacMurdockSzego")
    return( KacMurdockSzego(matrix.params$rho, n))
  if(matrix.type == "FracBrownMotion")
  {
    if(!('T' %in% names(matrix.params)))
      matrix.params$T <- 1
    return( FracBrownMotion(matrix.params$alpha, n, matrix.params$T))
  }
}


# Create a matrix with A_ij = \rho^{|i-j|} 
KacMurdockSzego <- function(rho, n)
{
  return(toeplitz(rho**(0:(n-1))))
}

# Create a covariance matrix for a discrete Fractional Brownian Motion (FBM) with B_t the FBM at time t
# Input: 
# H - Hurst parameter
# n - signal length 
# T - total signal time (default: T=1)
# Output: 
# A matrix C such that C_{ij} is the covariance of B_{(i+1)/n}-B_{i/n} and B_{(j+1)/n}-B_{j/n}
FracBrownMotion <- function(H, n, T = 1, plot.flag = FALSE)
{
  sigma <- 1
  delta.t <- T/n  # was 1/n
  top.vec <- 0:(n-1)
  top.vec <- sigma * delta.t**(2*H) * 
    (0.5*((abs(top.vec+1))**(2*H) + (abs(top.vec-1))**(2*H)) - top.vec**(2*H))
  if(plot.flag)
  {
    plot(0:(n-1), top.vec, type="l",
         main=paste0("H=", as.character(H)), xlab="Delta t", ylab="Corr.")
  }
  return(toeplitz(top.vec))
}

# Plot how the covariance changes vs. delay between two time points 
plot_covariance_vs_time <- function(matrix.type, scalar.params.vec, n, 
                                    force.run = TRUE, add.legend = TRUE, corr.flag = FALSE)
{
  num.params <- length(scalar.params.vec)
  print(num.params)
  col.vec <- get_color_vec("red", "blue", num.params)
  max.y <- min.y <- 0
  matrix.params <- c()
  for(i in 1:num.params)
  {
    print(paste0("Run param: ", as.character(scalar.params.vec[i])))
    matrix.params$rho = scalar.params.vec[i]
    matrix.params$alpha = scalar.params.vec[i]
    A <- build_covariance_matrix(matrix.type, matrix.params, n)
    cov.vec <- A[1,2:n]
    if(corr.flag)
    {
      y.lab <- "\\rho(B_i, B_j)"
      cov.vec = cov.vec * (n**(2*scalar.params.vec[i]))
    } else
      y.lab <- "COV(B_i, B_j)"
    max.y <- max(max.y, max(cov.vec))
    min.y <- min(min.y, min(cov.vec))
  }
  for(i in 1:num.params)
  {
    matrix.params$rho = scalar.params.vec[i]
    matrix.params$alpha = scalar.params.vec[i]
    A <- build_covariance_matrix(matrix.type, matrix.params, n)
    cov.vec <- A[1,2:n]
    if(corr.flag)
      cov.vec = cov.vec * (n**(2*scalar.params.vec[i]))
    if(i == 1)
      plot(1:(n-1), cov.vec, xlab="|i-j|", ylab=y.lab, col=col.vec[1], type="l",
           ylim=c(min.y, max.y), cex=1.25, lwd=2)
    else
      lines(1:(n-1), cov.vec, col=col.vec[i], cex=1.25, lwd=2)
  }
  if(add.legend)
    legend(n*0.7, 0.9*max.y, paste0(rep("H=", num.params), as.character(scalar.params.vec)), 
           col=col.vec[1:num.params],  lty=rep(1, num.params), 
           cex=1.25, box.lwd = 0, box.col = "white", bg = "white")
}


# Plot optimal protfolio value as function of the delay D
# Input: 
# matrix.type - string specifying the covariance matrix family
# matrix.params - fixed parameters for all matrices
# scalar.params.vec - vector specifying the parameter determining the covariance
# mu - mean of the increments
# D.vec - vector of discrete delay values
# plot.type - show curves for different delays or different parameters values
#
plot_protfolio_value <- function(matrix.type, matrix.params=c(), scalar.params.vec, mu, D.vec, 
                                 plot.type="by.H", force.run = TRUE, add.legend = TRUE)
{
  n <- length(mu)
  params.str <- c("\\rho", "H")
  names(params.str) <- c("KacMurdockSzego", "FracBrownMotion")
  value.file.name <- paste0("game_value_", matrix.type, as.character(n), ".RData")
  if(file.exists(value.file.name) && (force.run == FALSE))
  {
    load(value.file.name)
    num.params <- length(scalar.params.vec)
    num.D <- length(D.vec)
  }  else {
#    matrix.params <- c()
    num.params <- length(scalar.params.vec)
    num.D <- length(D.vec)
    n <- length(mu)
    value.mat <-  matrix(0, nrow=num.params, ncol=num.D) #  rep(0, length(D.vec))
    for(i in 1:num.params)
    {
      print(paste0("Run param: ", as.character(scalar.params.vec[i])))
      matrix.params$rho = scalar.params.vec[i]
      matrix.params$alpha = scalar.params.vec[i]
      A <- build_covariance_matrix(matrix.type, matrix.params, n)
      for(j in 1:num.D)
        value.mat[i,j] <- protfolio_value(A, mu, D.vec[j])
    }  
    save(value.mat, matrix.type, matrix.params, scalar.params.vec, mu, D.vec, file=value.file.name)
  }
  if(plot.type == "by.H")
  {
    col.vec <- get_color_vec("red", "blue", num.params)
    plot(D.vec, value.mat[1,], xlab="H", ylab="Value", col=col.vec[1], 
         ylim=range(value.mat), pch=20, cex=1.25, lwd=2)
    for(i in 2:num.params)
      points(D.vec, value.mat[i,], col=col.vec[i], pch=20, cex=1.25, lwd=2)
    if(add.legend)
      legend(max(D.vec)*0.85, max(value.mat)*0.9, as.character(scalar.params.vec), 
             col=col.vec[1:num.params],  pch=rep(20, num.params), 
             cex=1.25, box.lwd = 0, box.col = "white", bg = "white")
  } else # here each H is a color, plot vs. parameter
  {
    col.vec <- get_color_vec("red", "blue", num.D)
    par(mar=c(5,6,4,1)+.1)
    plot(scalar.params.vec, value.mat[,1], xlab=TeX(paste0("$", params.str[matrix.type], "$")), #    xlab=expression(params.str[matrix.type]), 
         ylab="Value", col=col.vec[1], ylim=range(value.mat), type="l", lwd=2, 
         cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2)
    for(j in 2:num.D)
      lines(scalar.params.vec, value.mat[,j], col=col.vec[j])
    if(add.legend)
      legend(-0.0428, max(value.mat)+0.0000000398, paste0(rep("D=", num.D), as.character(D.vec)),
             col=col.vec[1:num.D],  lty=rep(1, num.D), lwd=2,
             cex=1.9, box.lwd = 0, box.col = "white", bg = "white")
    grid(col = "darkgray", lwd=1.5)
  }
  return(value.mat)
}
  

# Create color map for plotting
get_color_vec <- function(col1 = "red", col2 = "blue", num.c=5)
{
  pal <- colorRamp(c(col1, col2))  # choose colors for two extremes
  c.col.vec <- matrix(0, num.c, 3) # rep('', num.k)
  for(k in c(1:num.c))
  {
    c.col.vec[k,] <- pal((k-1) / (num.c-1))
  }
  c.col.vec <- c.col.vec/ 255
  ret.c.col.vec <- rep("", num.c)
  for(k in c(1:num.c))
  {
    ret.c.col.vec[k] <- rgb(c.col.vec[k,1], c.col.vec[k,2], c.col.vec[k,3])
  }
  return(ret.c.col.vec)
}

# Plot a matrix as a heatmap with colormap 
heatmap_ggplot <- function(A)
{
  WD <- A %>% as.data.frame() %>%
    rownames_to_column("rows") %>%
    pivot_longer(-c(rows), names_to = "cols", values_to = "counts") 
    
    WD$rows <- as.integer(WD$rows)    
    WD$cols <- as.integer(gsub("V", "", WD$cols))
    
    WD %>% ggplot(aes(x=cols, y=rows, fill=counts)) + 
    geom_raster() + 
    scale_fill_continuous()
}

