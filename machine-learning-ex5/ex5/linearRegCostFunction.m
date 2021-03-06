function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% regularization term for linear regression cost function
reg_term = (lambda/(2 * m)) * sum(theta(2:end,:).^2);

predictions = X * theta;                       % predictions of hypothesis on all m examples
sqrErrors = (predictions - y).^2;              % square errors

% compute regularized linear regression cost function
J = (1/(2 * m)) * sum(sqrErrors) + reg_term;

% compute the unregularized linear regression gradient
grad = (1/m) * sum( (predictions - y) .* X);
temp = theta;
temp(1) = 0;                                   % because we don't regularize theta 0

grad = grad(:);
% add the regularization term to the 
grad = grad + ((lambda/m) * temp);

% =========================================================================

end
