function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

predictions = X * Theta';

diff = (predictions(R == 1) - Y(R == 1)) .* (predictions(R == 1) - Y(R == 1))
J = sum(diff) / 2 + (lambda / 2) * sum(sum(Theta .* Theta,2),1) + (lambda / 2) * sum(sum(X .* X,2),1);
for i = 1 : num_movies
  theta = Theta(find(R(i, :) == 1),:);
  x_i = X(i, :);
  y_i_j = Y(i, find(R(i, :) == 1));
  X_grad(i,:) = ((x_i * theta' - y_i_j) * theta) + lambda * x_i;
endfor
for j = 1 : num_users
  idx = find(R(:, j) == 1);
  theta = Theta(j, :);
  x_i = X(idx,:);
  y_i_j = Y(idx, j);
  Theta_grad(j,:) = ((x_i * theta' - y_i_j)' * x_i) + lambda * theta;
endfor


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
