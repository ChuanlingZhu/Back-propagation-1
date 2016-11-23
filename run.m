%This is a simple implementation with 3 layers i.e. one input layer, one
%hidden layer and one output layer. 
%Since this is for binary classification, we will have two input neurons
%and one output neuron. We will experiment with the number of neurons in
%the hidden layer.
function [computedOutput] = run(inputMat, targetVec, numNeuronsHiddenLayer, lambda, stepSize)

    %check supplied paramaters
    if(~isempty(inputMat) && ~isempty(targetVec) && numNeuronsHiddenLayer ~= 0)
        %Initializing some other variables
        A1 = ones(size(inputMat, 2)+1, 1);
        A2 = ones(numNeuronsHiddenLayer+1, 1);
        A3 = 0;
        ipSize = size(inputMat, 1);
        
        %Random initialization of weight vectors and cost function
        costFunctionJ = 999999;
        prevCostFunValue = 0;
        iterations = 1;
        
        %this is how this code works. We need to define the size of the
        %weightvector. If L1 is the first layer, L2 is the second layer and
        %L3 is the third layes then the size of the weight vector is
        %(L2*L1+1)+(L3*L2+1). This formulation can be extended to a neural
        %network of any size.
        weightVec = rand(numNeuronsHiddenLayer*3 + 1*(numNeuronsHiddenLayer+1), 1);
        
        Theta1 = reshape(weightVec(1:numNeuronsHiddenLayer*size(A1, 1), 1), [numNeuronsHiddenLayer, size(A1, 1)]);
        Theta2 = reshape(weightVec(numNeuronsHiddenLayer*size(A1, 1)+1:size(weightVec, 1), 1), [1, size(A2, 1)]);
        
        while iterations < 10000
            %Store value of computed output
            computedOutput = zeros(size(inputMat, 1), 1);
            b_delta1 = zeros(4, 3);
            b_delta2 = zeros(1, 5);
            %do this till the cost functions converge 
            for i=1:size(inputMat, 1)
              %call forward propagation
              % A1 is always equal to input with a bias value.
              A1(2:size(A1, 1), 1) = inputMat(i, :);

              %Now we need to find Z2. This is Theta1*A1. Theta1 is part of the
              %weightVec and can be easily obtained using reshape function
              Z2 = Theta1*A1;

              %to compute A2 we need to find A2 which is g(Z2) where each row in Z2 is a
              A2(2:size(A2, 1), 1) = sigmf(Z2, [1, 0]);

              %Calculate Z3
              Z3 = Theta2*A2;

              %A3 has no bias and is calculated directly. This is also what we
              %call output.
              A3 = sigmf(Z3, [1, 0]);
              computedOutput(i, 1) = A3;

              %call backward propagation
              %Find the value for s_delta for the last layer
              s_delta3 = A3 - targetVec(i, 1);

              %now use this value to compute s_delta2 and s_delta1
              s_delta2 = Theta2'.*s_delta3.*(A2.*(1-A2));
              b_delta1 = b_delta1 + s_delta2(2:size(s_delta2, 1))*A1';
              b_delta2 = b_delta2 + s_delta3*A2';
            end

            D1 = (1/size(inputMat, 1))*b_delta1 + lambda.*Theta1;
            D2 = (1/size(inputMat, 1))*b_delta2 + lambda.*Theta2;

            %time to update the Paramters
            Theta1 = Theta1 - stepSize.*D1;
            Theta2 = Theta2 - stepSize.*D2;

            %Cost function J
            prevCostFunValue = costFunctionJ;
            costFunctionJ = (-1/ipSize)*(log(computedOutput)'*targetVec + log(1-computedOutput)'*(1-targetVec)) + ((lambda/2*ipSize)* sum((sum((Theta1).^2) + sum((Theta2).^2))));
            iterations = iterations + 1;
        end
    else
        print('Invalid input');
    end
end