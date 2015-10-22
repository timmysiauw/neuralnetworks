classdef NN < handle 
   % Implementation of simple feed forward neural network (NN) from Andrew 
   % Ng's Coursera. No regularization. 
   %
   % Properties:
   %
   %    config:     array containing number of nodes in each layer.
   %                config(i) = number of nodes in layer_i NOT including 
   %                the bias node
   %                layer_1 is assumed to be the input layer
   %                layer_(last) is assumed to be the output layer
   %
   %    nLayers:    (for convenience) the number of layers in the network
   %
   %    layers:     struct array containing layers of network
   %                layer.A is an array of node activations NOT including
   %                bias
   %                layer.W is a matrix containing the outgoing weights 
   %                for layer. layer_(last) does not have weights. 
   %                layer.D is an array of errors (for back propagation).
   %                layer.dW is a matrix containing the partial derivatives
   %                of the cost function with respect to the weights.
   % 
   %
   % Methods:
   %    
   %    nn = NN(config):    constructor for NN class returns neural network
   %                        instance. 
   %
   %    yhat = forward(x):  returns prediction, yhat, for input vector x
   %                        (can be row or column). 
   %
   %    backward(X, Y, eta): for row wise inputs, X, and corresponding 
   %                        correct responses, Y, performs one back
   %                        propagation pass through the entire training
   %                        set with learning parameter, eta. 
   %
   %    train(X, Y, eta, iter): performs back propogation on training set
   %                        (X, Y) iter times. Future work will make this
   %                        function more sophisticated. 
   %
   %    plot():             creates visualization for network.
   %
   
   
    properties
        config = [];
        nLayers = 0;
        layers = [];
    end
    
    methods 
        
        function [nn] = NN(config)
            % constructor for NN class
            % config: array containing number of layer nodes
            
            nn.config = config;
            nn.nLayers = length(config);
            
            for i = 1:nn.nLayers
               
                if i < nn.nLayers
                    layer.W = randn(config(i+1), config(i)+1);
                else
                    layer.W = [];
                end
                
                layer.A = [];
                layer.D = [];
                layer.dW = zeros(size(layer.W));
                
                nn.layers = [nn.layers, layer];
     
            end
                        
        end % end NN
        
        function [yhat] = forward(nn, x)
            % forward propogation of input, x
            
            % set activation of first layer to inputs (with bias)
            nn.layers(1).A = [x(:); 1];
            
            % loop through remaining layers
            for i = 2:nn.nLayers
                
                if i ~= nn.nLayers
                    % for all but the last layer, the activation is the
                    % activation of the previous layer, weighted and 
                    % squashed by the sigmoid function, with the bias added.
                    nn.layers(i).A = [NN.sigmoid(nn.layers(i-1).W*nn.layers(i-1).A); 1];
                else
                    % for the last layer, the activation is just previous
                    % weighted and squashed by the sigmoid function,
                    % without the bias added. 
                    nn.layers(i).A = NN.sigmoid(nn.layers(i-1).W*nn.layers(i-1).A);
                end
                
            end
            
            % prediction is the last layer's activation
            yhat = nn.layers(end).A;
            
        end % end forward
        
        function backward(nn, X, Y, eta)
            % backward propogation through training set (X, Y), with
            % learning parameter, eta
           
            % get number of inputs in this training set
            m = size(X,1);
                        
            % loop through entire training set
            for i = 1:m
            
                % get i'th candidate
                x = X(i,:);
                y = Y(i,:);
                
                % forward propogate to get approximation (and activations)
                yhat = nn.forward(x(:));

                % get deltas of approximation at output layer
                % NOTE: 0 is artificially appended to the delta here. The
                % reasoning is as follows. Andrew Ng's notes say that you
                % must use the entire delta vector when computing dW.
                % However, this doesn't account for bias nodes. When there
                % are bias nodes, the delta for that node must be omitted
                % because there isn't any error associated their activation
                % value (1). To account for this, the following for-loop
                % omits the last element of layer(i).D, which is associated
                % with the bias node for layer_i. To make the code
                % consistent with the first D value, an extra 0 is
                % appended - rather than accounting for it with an
                % if-statement in the for loop. 
                nn.layers(end).D = [yhat - y(:); 0];

                for j = nn.nLayers-1:-1:1
                    % from the notes:
                    
                    % D^(el) = W^(el)T*d^(el) .* a^(el).*(1 - a^(el))
                    nn.layers(j).D = (nn.layers(j).W'*nn.layers(j+1).D(1:end-1)).*(nn.layers(j).A.*(1 - nn.layers(j).A));
                    
                    % del^(el) = del^(el) + d^(el)*a^(el)T
                    nn.layers(j).dW = nn.layers(j).dW + nn.layers(j+1).D(1:end-1)*nn.layers(j).A';
                end
            
            end
            
            for i = 1:nn.nLayers-1
                % update the weights of each layer
                nn.layers(i).W = nn.layers(i).W - eta*nn.layers(i).dW/m;
                % reset dW to zeros
                nn.layers(i).dW = zeros(size(nn.layers(i).W));
            end
                        
        end % end backward
        
        function train(nn, X, Y, eta, iter)
            % run back propagation algorithm iter times over training set
            % (X, Y) with learning parameter, eta. 
            
            for i = 1:iter
                disp(sprintf('Iteration %d', i))
                nn.backward(X, Y, eta);
            end
        end % end train
        
        function plot(nn)
            % plot network
            
            figure
            
            set(gcf, 'Color', 'w')
            
            hold on
            
            for i = 1:nn.nLayers
             
                if i ~= nn.nLayers
                    for j = 1:size(nn.layers(i).W,2)
                       for k = 1:size(nn.layers(i).W,1)
                           p1 = [i, j];
                           p2 = [(i+1), k];
                           if rem(j,2) == 0
                               pm = p1 + (1/3)*(p2 - p1);
                           else
                               pm = p1 + (2/3)*(p2 - p1);
                           end
                           plot([p1(1), p2(1)], [p1(2), p2(2)], 'k', 'LineWidth', 3)
                           text(pm(1), pm(2), sprintf('%d', round(nn.layers(i).W(k,j))), 'Color', 'r', 'HorizontalAlignment', 'center', 'FontSize', 18)
                       end
                    end
                end
                
                if i == 1
                    color = 'b';
                elseif i == nn.nLayers
                    color = 'm';
                else
                    color = 'w';
                end
                
                n = nn.config(i);
                
                plot(i*ones(1,n), 1:n, 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', 'k', 'MarkerSize', 20)
                
                if i ~= nn.nLayers
                    plot(i, n+1, 'o', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'MarkerSize', 20)
                end
            end
            
            axis equal
            axis off
            
        end % end plot
        
    end % end methods
    
    methods (Static)
        
        function [s] = sigmoid(z)
            s = 1./(1 + exp(-z));
        end
        
    end % end static methods
    
end