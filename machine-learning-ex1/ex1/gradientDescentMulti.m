function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
sums = zeros(size(X,2), 1);                      % initialize sums and temp to column size of dataset X
temp = zeros(size(X,2), 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    predictions = X * theta;                         % predictions of hypothesis on all m examples
    
    % gradient descent for multi-variant regression problem
    for i=1:length(sums)
        sums(i, 1) = sum((predictions - y) .* X(:,i));  % sum for theta i
        temp(i, 1) = theta(i, 1) - ((alpha / m) * sums(i, 1)); % calculate gradient descent for theta i
    end
    theta = temp;                                    % update theta





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
