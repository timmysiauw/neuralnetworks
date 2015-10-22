% Learning to identify exactly 2 out of 3 ones with simple feed-forward 
% neural network
% notes: required a hidden layer with 3 nodes (it seems)
clc
clear
close all

nn = NN([3 3 1]);

X = [0 0 0;
    0 0 1;
    0 1 0;
    0 1 1;
    1 0 0;
    1 0 1;
    1 1 0;
    1 1 1];

Y = [0; 0; 0; 1; 0; 1; 1; 0];

nn.train(X, Y, 1, 10000);

for i = 1:size(X,1)
    disp(sprintf('Test #%d - response: %d - should be: %d', i, round(nn.forward(X(i,:))), Y(i)))
end

nn.plot()