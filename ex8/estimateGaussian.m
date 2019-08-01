function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% Your task is to complete the code in estimateGaussian.m. This function
%takes as input the data matrix X and should output an n-dimension vector
%mu that holds the mean of all the n features and another n-dimension vector
%sigma2 that holds the variances of all the features. You can implement this
%using a for-loop over every feature and every training example (though a
%vectorized implementation might be more efficient; feel free to use a vector-
%ized implementation if you prefer).

% By default, mean takes column mean 
% Transpose to row vector
mu = mean(X)';


% One feature (col) at a time, considering all samples
% X - mu: reduce col mean

% Mu back to row vector
% Sum takes colsums by default
sigma2 = ((1/m) * sum((X - mu').^2))';

% X is m x n


% =============================================================


end
