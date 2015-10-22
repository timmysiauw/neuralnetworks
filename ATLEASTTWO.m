% Learning to identify at least 2 out of 3 ones with simple feed-forward 
% neural network
% notes: only required a two-layer network
clc
clear
close all

nn = NN([3 1]);

X = [0 0 0;
    0 0 1;
    0 1 0;
    0 1 1;
    1 0 0;
    1 0 1;
    1 1 0;
    1 1 1];

Y = [0; 0; 0; 1; 0; 1; 1; 1];

nn.train(X, Y, 1, 1000);

for i = 1:size(X,1)
    disp(sprintf('Test #%d - response: %d - should be: %d', i, round(nn.forward(X(i,:))), Y(i)))
end

nn.plot()