function [wji, wkj, rss] = backpropagation_number(wji,wkj,data_x,data_t)

% data_x: input, column vectors
% data_t: target

n = size(data_x,2); % data_x: 2 by 4 --> n = 4;
eta = 1;
z_set = [];
for p=1:n
    % data
    x = data_x(:,p);
    t = data_t(:,p);
    
    % feedforward
    y = sigmoid_func(wji*x);
    z = softmax_func(wkj*y);
    
    % backpropagation
    delta_k = (t-z).*z.*(1-z);
    delta_j = (y.*(1-y)).*(wkj'*delta_k);
    
    % update weights    
    wji = wji + eta*delta_j*x';
    wkj = wkj + eta*delta_k*y';
    % feedforward
    x = data_x(:,p);
    y = sigmoid_func(wji*x);
    z = softmax_func(wkj*y);
    % save results
    z_set = [z_set,z];
    
end
rss = norm(z_set - data_t).^2;

