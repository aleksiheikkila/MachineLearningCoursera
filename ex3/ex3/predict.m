function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% X contains examples (images) in rows

% The matrices Theta1 and Theta2 contain the parameters for each unit in rows.
% Specifically, the first row of Theta1 corresponds to the first hidden unit in the second layer.

% Useful values
m = size(X, 1);  % nbr of images
num_labels = size(Theta2, 1);  % nbr of classes

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);  % row vector, p = class for each image


% In the case of the given ex:
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add column of zeros to X (the bias unit)
X = [ones(m, 1) X];

z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];


z3 = a2 * Theta2';
h = sigmoid(z3);  % Output dims: rows = nbr of images, cols = nbr of classes

% From each row, pick max 
[maxval p] = max(h, [], 2); 



% =========================================================================


end
