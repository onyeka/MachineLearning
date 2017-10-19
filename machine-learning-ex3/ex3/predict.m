function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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

% Add a column of ones representing the x0 feature to X
X = [ones(m, 1) X];

% generate hidden layer
A1 = sigmoid(X * Theta1');
fprintf('\n A1: [%d x %d]\n', size(A1));

% Add a column of ones representing the A1,0 bias unit to A1 
A1 = [ones(m, 1) A1];

% generate output layer
A2 = sigmoid(A1 * Theta2');
fprintf('\n A2: [%d x %d]\n', size(A2));

% get the max for each row and the class it represents.
% The class is the index into the row where the max occurs
[mx, idx] = max(A2, [], 2);
fprintf('\n mx: [%d x %d], idx: [%d x %d]\n', size(mx), size(idx));

p = idx;






% =========================================================================


end
