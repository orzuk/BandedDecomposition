# Banded Decomposition
Matrix decomposition for investing with delay in assets with price following Gaussian processes, with dynamic trading and 
also including a static option, i.e. semi-static hedging. 

This repository contains R code used to calculate the value of an investor following an optimal policy with delay, 
for a market with increments having a known joint Gaussian distribution. 
The optimal investment policy and the value attained are obtained by a banded matrix decomposition of the inverse covaraince matrix [1].
In addition, the optimal investment policy and the value attained for a semi-static problem are obtained for the Gaussian i.i.d. case [2].
The repository accompanies the papers [1] and [2]. Please cite these papers if you're using the package. 

## Installation
Clone the repository into a directory of your choice, and start an R session within this directory. 
Change the main path using 'setwd' to the path where the repository is located in your computer. 

## Usage examples 
Run the script 'RunBandedDecomposition' to plot the value as a function of the delay and process parameters for 
two examples (Kac-Murdock-Szego and Fractional-Brownian-Motion covariance matrices) shown in [1]. 

Run the script 'RunSemiStatic' to plot the value of semi-static hedging as a function of the delay and process parameters for an example
shown in [2].

### Authors
For any questions, please contact Yan Dolinsky (yan.dolinsky@mail.huji.ac.il) or Or Zuk (or.zuk@mail.huji.ac.il)


### Ref
[1] [Exponential Utility Maximization in a Discrete Time Gaussian Framework](https://arxiv.org/abs/2305.18136) <br>
Y. Dolinsky and O. Zuk, arXiv (2023)<br>

[2] [Explicit Computations for Delayed Semistatic Hedging](https://arxiv.org/abs/2308.10550) <br>
Y. Dolinsky and O. Zuk, arXiv (2023)<br>
 
