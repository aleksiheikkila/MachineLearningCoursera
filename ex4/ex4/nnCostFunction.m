function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);  % training examples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add column of zeros to input X (the bias unit)
%X = [ones(m, 1) X];   % row = one sample
% X now 5000 x 401
% Theta1 = 25 x 401

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';   % Practically: Input to the hidden layer units. Thetat: rivi = yhden unitin asiat. K‰‰nnet‰‰n sarakkeeksi
a2 = sigmoid(z2);   % Outputs from the hidden layer units
% 5000 x 25
a2 = [ones(size(a2, 1), 1), a2];
% 5000 x 26

z3 = a2 * Theta2';  % Inputs to the output layer units. Theta2 = 10 x 26
h = sigmoid(z3);  
% 5000 x 10


% Then the cost function (without regu)
% Loop through the training examples
for i = 1:m,     % 1 to 5000
  yVec = zeros(1, num_labels);  % rivivektori
  yVec(y(i)) = 1;
    
  % Add the cost func
  J = J + sum(( -yVec .* log(h(i,:)) - (1-yVec) .* log(1 - h(i,:)) ));
  
  % Backpropagagate
  % Errors at output. a2 = 5000 x 26
  d3 = h(i,:) - yVec;  % 1x10


  % d2 = Theta2' * d3' .* sigmoidGradient(a2(i,:))';  % 26 x 1
  d2 = Theta2' * d3' .* sigmoidGradient([1, z2(i,:)])';   % !z2 but had to add the bias "1" to the beginning!
  d2 = d2(2:end);  % 25x1
  
  % Deltas. Used in the end to calc the grads
  
  % Theta in matrix format
  % l = layer, i = training example, j = node.
  
  % l = 2. i loop variable. j = node
  % Theta2_grad = 10 x 26. d3 = 1x10, a2 = 5000 x 26, a2(i,:) = 1x26
  Theta2_grad = Theta2_grad + ( d3' * a2(i,:) );
  
  % l = 1. i loop variable. j = node
  % Theta1_grad = 25 x 401
  
  % a1 = 1 x 401, d2 = 26x1, d2 modified 25x1, a1 = x = 1x401
  Theta1_grad = Theta1_grad + ( d2 * a1(i,:) );  % a1 = x
  
end

J = (1/m) * J;  % This is the cost function value for theta params

% add regularization to your cost function
% Do not regularize  the terms that correspond to the bias (first columns in Theta matrices)
J = J + (lambda/(2*m)) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );


% Obtain the (unregularized) gradient for the neural network cost function
Theta2_grad = (1/m) .* Theta2_grad;
Theta1_grad = (1/m) .* Theta1_grad;

% Regularization added (not for the bias terms)
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);

% Unroll to grad
grad = [Theta1_grad(:); Theta2_grad(:)];


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
