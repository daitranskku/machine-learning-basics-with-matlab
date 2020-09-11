function y = softmax_func(x)
    exp_x = exp(x);
    y = exp_x/ sum(exp_x);
end