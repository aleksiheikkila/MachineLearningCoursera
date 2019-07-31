function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));    % montako featurea. rivivektori sen mukaan
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%

% Mean function:  If X is a matrix, compute the mean for each column and return them in a row vector. 
% Same for std function
mu = mean(X);
sigma =  std(X);

% X = m x n
% mu ja sigma 1 x n
X_norm = (X - mu) ./ sigma;

% Tis is the automatic broadcasting feature (in newer Octave?)
% http://math.stackexchange.com/questions/86848/to-subtract-two-matrices-with-different-dimensions-in-octave-matlab


% ============================================================

end
