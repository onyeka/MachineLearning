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
m = size(X, 1);
         
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

% add the column of bias units to the examples
X = [ones(m, 1) X];

% compute the hidden layer activation units of the neural network [5000 x 25]
[hidden_layer] = sigmoid(X * Theta1');

% add the column of bias units to the hidden layers
hidden_layer = [ones(size(hidden_layer, 1), 1) hidden_layer];

% compute the output layer activation units of the neural network [5000 x 10]
[output_layer] = sigmoid(hidden_layer * Theta2');

% compute the cost function of the neural network
yv = [1:num_labels] == y;
J = (1/m) * sum( sum( (-yv .* log(output_layer)) - ( (1 - yv) .* log(1 - output_layer)) ) );

% compute the regularization element of the cost function
reg_theta1 = sum(sum(Theta1(:, 2:end).^2));
reg_theta2 = sum(sum(Theta2(:, 2:end).^2));
reg_cost   = (lambda/(2*m)) * (reg_theta1 + reg_theta2);


% compute the regularized cost function of the neural network
J = J + reg_cost;


% compute backpropagation
for t=1:m,
    a_1 = X(t, :);                      % assign activation units for input layer [1 x 401]
    z_2 = a_1 * Theta1';                % compute the z vector of the hidden layer [1 x 25]
    z_2 = z_2(:);                       % convert row vector to column vector [25 x 1]
    a_2 = sigmoid(z_2);                 % compute activation units for hidden layer

    a_2 = [1; a_2];                     % add bias unit to activation units [26 x 1]
    z_3 = Theta2 * a_2;                 % compute the z vector of the output layer [10 x 1]
    a_3 = sigmoid(z_3);                 % compute activation units for output layer [10 x 1]

    L_3 = size(a_3, 1);                 % number of units in output layer [10]
    d_3 = zeros(size(a_3));             % error of node in output layer [10 x 1]

    % compute the error in the output layer using back propagation
    d_3 = a_3 - yv(t, :)';

    % compute the error in the hidden layer using back propagation
    tmp = Theta2' * d_3;
    d_2 = tmp(2:end) .* sigmoidGradient(z_2); % error of node in hidden layer [25x1]

    % accumulate the errors to compute the gradients
    Theta2_grad = Theta2_grad + d_3 * a_2';
    Theta1_grad = Theta1_grad + d_2 * a_1;
end

% compute regularized component of the gradient
reg_Theta1_grad = (lambda/m) * Theta1;
reg_Theta2_grad = (lambda/m) * Theta2;
reg_Theta1_grad(:, 1) = 0;
reg_Theta2_grad(:, 1) = 0;

% compute the regularized gradients for the neural network
Theta1_grad = Theta1_grad/m + reg_Theta1_grad;
Theta2_grad = Theta2_grad/m + reg_Theta2_grad;











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
