function y = testCase3_obj(x)
    n = length(x);
    sum1 = 0;
    sum2 = 0;
    for i = 1:n
        sum1 = sum1 + x(i)^2;
        sum2 = sum2 + cos(2*pi*x(i));
    end
    term1 = -20 * exp(-0.2 * sqrt(sum1/n));
    term2 = -exp(sum2/n);
    y = term1 + term2 + 20 + exp(1);
end
