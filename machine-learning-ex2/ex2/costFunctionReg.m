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
regCost=0;

for j=2:size(theta),
  regCost=regCost+theta(j).*theta(j);
end
regCost=regCost*lambda/2/m;
for i=1:m,
  cost=(-y(i))*log(sigmoid(dot(theta',X(i,:)))) - ((1-y(i)) * log(1- (sigmoid(dot(theta',X(i,:))))));
  J=J+cost; 
end
J=J/m;
J=J+regCost;

for k=1:m,
    cost=(sigmoid(dot(theta',X(k,:)))-y(k))*X(k,1);
    grad(1)=grad(1)+cost;
  end
  grad(1)=grad(1)/m;

for j=2:size(theta)
  for k=1:m,
    cost=(sigmoid(dot(theta',X(k,:)))-y(k))*X(k,j);
    grad(j)=grad(j)+cost;
  end
  grad(j)=grad(j)/m;
  grad(j)=grad(j)+lambda*(theta(j))/m;
end

% =============================================================

end
