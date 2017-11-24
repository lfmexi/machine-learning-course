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

X = [ones(m, 1) X];

Z2 = X * Theta1';
A2 = [ones(size(Z2),1) sigmoid(Z2)];

Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

for i = 1:m,
  exampleY = zeros(1, num_labels);
  exampleY(1, y(i)) = 1;
  
  for k = 1:num_labels,
    costTerm = -exampleY(k)*log(A3(i, k)) - (1 - exampleY(k))* log(1 - A3(i, k));
    J += costTerm;
  end;
end;

D1 = 0;
D2 = 0;

for t = 1:m,
  exampleY = zeros(1, num_labels);
  exampleY(1, y(t)) = 1;

  delta3 = A3(t, :) - exampleY;
  delta3 = delta3';
  delta2 = Theta2'(2:end,:) * delta3 .* sigmoidGradient(Z2(t, :)');

  D1 = D1 + delta2 * X(t,:);
  D2 = D2 + delta3 * A2(t, :);
end

squaredTheta1 = Theta1(1:end, 2:end) .^ 2;

squaredTheta2 = Theta2(1:end, 2:end) .^ 2;

flatSquaredTheta1 = squaredTheta1(:);
flatSquaredTheta2 = squaredTheta2(:);

regularizationTerm = sum(flatSquaredTheta1) + sum(flatSquaredTheta2);

J = J/m;

J = J + (lambda/ (2 * m)) * regularizationTerm;

Theta1_grad = (1 / m) * D1 + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end) ];
Theta2_grad = (1 / m) * D2 + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end) ];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
