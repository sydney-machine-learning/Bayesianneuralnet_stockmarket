# Bayesian neural network  with Parallel Tempering MCMC for Stock Market Prediction
\Overview
An experimental project under Bayesian neural networks given the use of Langevin gradients with parallel tempering MCMC.

The proposal here is to compare our forecasting model with state-of-art neural network  training algorithms(FNN-SGD and FNN-Adam)

data.py is used for data preprocessing.

To run the results, desired parameters should be set in nn.py

Following are some example results of MMM’s stock price prediction. They are They are one-step, two-step and five-step prediction result respectively. The grey area is the uncertainty of the prediction results.   
 
![image](https://user-images.githubusercontent.com/85796527/122184025-65164b00-cebe-11eb-97be-99842e910e36.png)
![image](https://user-images.githubusercontent.com/85796527/122184030-6778a500-cebe-11eb-805d-073f3fe8fb64.png)
![image](https://user-images.githubusercontent.com/85796527/122184037-69daff00-cebe-11eb-835d-d1132cabb630.png)



 

Also there is error analysis.
 
![image](https://user-images.githubusercontent.com/85796527/122184061-6e9fb300-cebe-11eb-9068-69aa3d87f66e.png)

Published research studies
