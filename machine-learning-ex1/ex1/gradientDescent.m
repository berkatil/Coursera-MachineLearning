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
    theta0=theta(1);
    theta1=theta(2);
    currentCost1=0;
    for i=1:m,
      temp=(theta')*[1; X(i,2)];
      temp=temp-y(i);
      temp=temp*X(i,1);
      currentCost1=currentCost1+temp;
    end
    currentCost1=currentCost1*alpha/m;
 #  theta(1)=theta0-currentCost1;
   
   currentCost=0;
    for i=1:m,
      temp=(theta')*[1; X(i,2)];
      temp=temp-y(i);
      temp=temp*X(i,2);
      currentCost=currentCost+temp;
    end
    currentCost=currentCost*alpha/m;
    theta(1)=theta0-currentCost1;
    theta(2)=theta1-currentCost;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
