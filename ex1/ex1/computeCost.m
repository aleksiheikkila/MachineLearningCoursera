function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% X matrix m x (n+1) (n features, +1 intercept). m = num of training examples


% Now: theta 2 x 1. Theta' 1 x 2
% X m x 2

h = X * theta;

J = (1/(2*m)) * sum( (h - y).^2 );  % .^ = elementwise


% =========================================================================

end
