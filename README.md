# BandedDecomposition
Matrix decomposition for investing with delay in assets with price following Gaussian processes

This repository contains R code used to calculate the value of an investor following an optimal policy with delay, 
for a market with increments having a known joint Gaussian distribution. 
The optimal investment policy and the value attained are obtained by a banded matrix decomposition of the inverse covaraince matrix.
The repository accompanies the paper [1]. Please cite this paper if you're using the package. 

## Installation
clone the repository into a directory of your choice, and start an R session within this directory. 
Change the main path using 'setwd' to the path where the repository is located in your computer. 

## Usage example 
Run the script 'RunBandedDecomposition' to plot the value as a function of the delay and process parameters for 
two examples (Kac-Murdock-Szego and Fractional-Brownian-Motion covariance matrices) shown in the paper. 


### Authors
For any questions, please contact Yan Dolinsky (yan.dolinsky@mail.huji.ac.il) or Or Zuk (or.zuk@mail.huji.ac.il)


### Ref
[1] Exponential Utility Maximization Problem in a Discrete Time Gaussian Framework <br>
Y. Dolinsky and O. Zuk, arXiv (2023)<br>
 
