function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
% order to use an off-the-shelf minimizer such as fmincg, the cost function has
% been set up to unroll the parameters into a single vector params.
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features. Yksi rivi on yhden userin
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
% Theta' = num_features x num_users
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% only has elements with values either 0 or 1, this has the
% effect of setting the elements of M to 0 only when the corresponding value
% in R is 0. Hence, sum(sum(R.*M)) is the sum of all the elements of M for
% which the corresponding element in R equals 1


% If there really is score, use it. Otherwise zero:
TrueScores = R .* Y;  % num_movies x num_users
%M = M';  % num_users x num_movies

% Predictions only for true scores (others not consider by the cost function)
PredScores = Theta * X';  % num_users x num_movies
PredScores = PredScores'; % num_movies  x num_users
PredScores = R .* PredScores;  % Just for true scores, otherwise zero

% Movie i, user j
% Cost funct without regu
%J = 0.5 * sum(sum((PredScores - TrueScores).^2));

% With regu
J = 0.5 * sum(sum((PredScores - TrueScores).^2)) + 0.5*lambda*( sum(sum(Theta .^2)) + sum(sum(X .^2)) );


% Gradients
% Without regu
%X_grad = (PredScores - TrueScores) * Theta;  
% Theta_grad = (PredScores - TrueScores)' * X;

% With regu:
X_grad = (PredScores - TrueScores) * Theta + lambda * X;  
% num_movies x num_users  *  num_users  x num_features  --> num_movies x num_features, seems like right dims!

Theta_grad = (PredScores - TrueScores)' * X + lambda * Theta;
% num_movies x num_users (transpose: num_users ' num_movies)  *  num_movies  x num_features  --> num_users x num_features, seems like right dims!


% =============================================================

% Roll the grad
grad = [X_grad(:); Theta_grad(:)];

end
