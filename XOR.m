% Learning XOR with simple feed-forward neural network
% notes: required one hidden layer (it seems)
clc
clear
close all

nn = NN([2 2 1]);

X = [0 0;
    0 1;
    1 0;
    1 1];

Y = [0; 1; 1; 0];

nn.train(X, Y, 1, 1000);

disp('XOR')
for i = 1:size(X,1)
    disp(sprintf('Test #%d - response: %d - should be: %d', i, round(nn.forward(X(i,:))), Y(i)))
end

nn.plot()