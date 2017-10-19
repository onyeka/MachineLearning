function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    predictions = X * theta;                   % predictions of hypothesis on all m examples
    sum0 = sum((predictions - y) .* X(:,1));   % sum for theta 0 
    sum1 = sum((predictions - y) .* X(:,2));   % sum for theta 1 

    temp0 = theta(1,:) - ((alpha / m) * sum0); % calculate gradient descent for theta 0
    temp1 = theta(2,:) - ((alpha / m) * sum1); % calculate gradient descent for theta 1
    theta = [temp0; temp1];                    % update theta







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
