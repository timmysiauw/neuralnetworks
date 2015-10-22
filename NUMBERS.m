% Learning to identify numbers from LeNet data set: http://yann.lecun.com/exdb/lenet/index.html
% notes: can't seem to get it to work
clc
clear
close all

load numbers.mat

% parameters and useful values
m = length(V);
training = 400; % how many of each number should we take

% expand correct values into binary array
Y = zeros(m,10);
for i = 1:m
   Y(i, V(i)) = 1; 
end

% turn inputs into binary (rather than signal values)
X = round(X);
X(X<0) = 0;
X(X>1) = 1;

% create training set
Xt = X(mod(1:m, 500)<training,:);
Yt = Y(mod(1:m, 500)<training,:);

% create neural network
nn = NN([400, 400, 400, 64, 10]);

% do training
nn.train(Xt, Yt, 1, 400);

% create test set
Xtest = X(mod(1:m, 500)>=training,:);
Ytest = Y(mod(1:m, 500)>=training,:);

% score test set
scores = zeros(1,10);

for i = 1:size(Xtest,1)
   
    [m, vhat] = max(nn.forward(Xtest(i,:)));
    
    if vhat == find(Ytest(i,:))
        scores(vhat) = scores(vhat) + 1;
    end
    
end

% display number of numbers correctly identified
for i = 1:10
    
    if i == 0
        n = 0;
    else
        n = i;
    end
    
    disp(sprintf('%d''s correctly identified: %d', n, scores(i)))

end

% see distribution of guess "weight"
H = zeros(10,10);
for i = 1:size(Xtest,1)
    H(Ytest(i,:)==1,:) = H(Ytest(i,:)==1,:) + nn.forward(Xtest(i,:))';
end

bar(H')
legend({'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'})