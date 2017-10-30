function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
x1 = [5 2 4];
x2 = [0 1 -1];

params = [0.01 0.03 0.1 0.3 1 3 10 30];
p_size = size(params, 2);
min_error = 1000;

% train SVM models based on guesses for C and sigma to
% determine the best C and sigma values for the cross evaluation set
%fprintf('Xval:[%dx%d], yval:[%dx%d], p_size: %d\n', size(Xval), size(yval), p_size);
%for i = 1:p_size,
%    for j = 1: p_size,
%        C_tmp = params(i);
%        sigma_tmp = params(j);
%        model = svmTrain(X,y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, sigma_tmp));
%        visualizeBoundaryLinear(X, y, model);
%        [predictions] = svmPredict(model, Xval);
%        err = mean(double(predictions ~= yval));
%        printf('err: %f, C: %f, sigma: %f\n', err, C_tmp, sigma_tmp);
%        if min_error > err,
%            min_error = err;
%            C = C_tmp;
%            sigma = sigma_tmp;
%            printf('!!!!! Choice Value C: %f, sigma: %f\n', C_tmp, sigma_tmp);
%        end
%    end
%end
%fprintf('C: %f sigma: %f\n', C, sigma);

% selected values based on lowest error returned
C = 1;
sigma = 0.1;





% =========================================================================

end
