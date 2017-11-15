function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
  m = length(y); % number of training examples

% You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

  function regTheta = regularizeTheta(t)
    regTheta = zeros(size(t));
    regTheta = regTheta + t;
    regTheta(1) = 0;
  end

  logisticCost = 0;

  for i = 1:m,
    x_j = X'(:,i);
    linearHyp = theta' * x_j;
    hypotesis = sigmoid(linearHyp);

    logisticCost = logisticCost + -1 * y(i) * log(hypotesis) - (1 - y(i)) * log(1 - hypotesis);

    nextMinSum = (hypotesis - y(i)) * x_j;
    grad = grad + nextMinSum + (lambda / m) * regularizeTheta(theta);
  end

  squareTheta = theta .^ 2;

  regularizationParam = (lambda / (2 * m)) * sum(squareTheta(2:size(squareTheta)));

  J = (1 / m) * logisticCost + regularizationParam;
  grad = (1 / m) * grad;
% =============================================================

end
