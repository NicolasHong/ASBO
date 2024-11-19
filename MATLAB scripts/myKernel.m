function G = myKernel(U,V)
% Define custom kernel
    % Get global variables
    global var1 var2;

    [m, ~] = size(U);
    [n, ~] = size(V);
    G = ones([m,n]);

    G = var1.*var2.*G;
end