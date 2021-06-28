# Bayesian neural network  with Parallel Tempering MCMC for Stock Market Prediction

## Overview

An experimental project under Bayesian neural networks using Langevin-gradients parallel tempering MCMC [Chandra et al,2019] which could be implemented in a parallel computing environment.

The proposal here is to compare our stock price forecasting model with state-of-art neural network training algorithms (FNN-SGD and FNN-Adam)

- data.py - This file is used for data preprocessing.

- nn.py - To run the results, desired parameters should be set in this file


## Sample Output

Following are some example results of MMM’s stock price prediction. They are They are one-step, two-step, five-step prediction result and error analysis respectively. The grey area is the uncertainty of the prediction results. 
 
![image](https://user-images.githubusercontent.com/85796527/122184025-65164b00-cebe-11eb-97be-99842e910e36.png)
![image](https://user-images.githubusercontent.com/85796527/122184030-6778a500-cebe-11eb-805d-073f3fe8fb64.png)
![image](https://user-images.githubusercontent.com/85796527/122184037-69daff00-cebe-11eb-835d-d1132cabb630.png)
![image](https://user-images.githubusercontent.com/85796527/122184061-6e9fb300-cebe-11eb-9068-69aa3d87f66e.png)


 
## Published research studies
- Chandra R ,  Jain K ,  Deo R V , et al. [Langevin-gradient parallel tempering for Bayesian neural learning](https://www.sciencedirect.com/science/article/abs/pii/S0925231219308069)[J]. Neurocomputing, 2019, 359(SEP.24):315-326.

When you use Bayesian neural network  with Parallel Tempering MCMC, please cite the above papers.
